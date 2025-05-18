from __future__ import annotations
import sys
import numpy as np
import numpy.typing as npt
import pathlib
from mcpc16 import Register, Condition, Operation, encode_instruction, PROGRAM_MEMORY_SIZE, inverse_condition
import argparse
from dataclasses import dataclass

GENERATED_UNIT_NAME = "__generated__"
START_LABEL_NAME = "global:start"
GLOBAL_SCOPE_NAME = "__global__"

ASSEMBLER_MACROS = {
    "true": "1",
    "false": "0",
    "c1": "r14",
    "c2": "r15",
    "r0": "pc",
}

@dataclass
class AssemblyScope():
    id: int
    name: str

    parent: AssemblyScope | None # Parent scope, None for the GLOBAL SCOPE.
    children: list[AssemblyScope]
    
    start: int # First instruction of the scope.
    end: int # Last instruction (exclusive) of the scope.

    labels: dict[str, AssemblyLabel]
    macros: dict[str, AssemblyMacro]

    def create_label(self, name: str, location: int, source: AssemblySourceLine | None) -> AssemblyLabel:
        # Check that label is unique.
        existing_label = self.find_label(name)
        if existing_label is not None:
            existing_source = existing_label.source
            if existing_source is not None:
                raise AssemblyError(instruction.source, f"Trying to redefine label '{name}' which is already defined in a visible scope in line {existing_source.line} of unit '{existing_source.unit}'.")
            else:
                raise AssemblyError(instruction.source, f"Trying to redefine label '{name}' which is already defined in a visible scope in the default labels.")

        # Define label
        label = AssemblyLabel(name, location, self, source)
        self.labels[name] = label
        return label
    
    def create_macro(self, name: str, value: str, source: AssemblySourceLine | None) -> AssemblyMacro:
        # Check that macro is unique in the current scope.
        existing_macro = self.macros.get(name)
        if existing_macro is not None:
            existing_source = existing_macro.source
            if existing_source is not None:
                raise AssemblyError(instruction.source, f"Trying to redefine macro '{name}' which is already defined in a visible scope in line {existing_source.line} of unit '{existing_source.unit}'.")
            else:
                raise AssemblyError(instruction.source, f"Trying to redefine macro '{name}' which is already defined in a visible scope in the default macros.")

        # Define macro
        label = AssemblyMacro(name, value, self, source)
        self.macros[name] = label
        return label

    def visible_labels(self) -> dict[str, AssemblyLabel]:
        # Add local labels.
        result = self.labels.copy()

        # Add child labels.
        for child_scope in self.children:
            result |= child_scope.visible_labels()
        
        # Add parent labels.
        next_parent = self.parent
        while next_parent is not None:
            result |= next_parent.labels
            next_parent = next_parent.parent

        # Return result.
        return result
    
    def find_label(self, name: str) -> AssemblyLabel | None:
        # Check local labels.
        if name in self.labels:
            return self.labels[name]
            
        # Check parents.
        next_parent = self.parent
        while next_parent is not None:
            if name in next_parent.labels:
                return next_parent.labels[name]
            next_parent = next_parent.parent

        # Check children.
        for child_scope in self.children:
            label = child_scope.find_label(name)
            if label is not None:
                return label

        # Nothing found.
        return None 

    def visible_macros(self) -> dict[str, AssemblyMacro]:
        # Add local macros.
        result = self.macros.copy()
        
        # Add parent macros.
        next_parent = self.parent
        while next_parent is not None:
            # Add macros from parent. The order here is important, the rhs takes priority.
            # In this order this means that "higher" scopes take priority.
            result = next_parent.macros | result
            next_parent = next_parent.parent

        # Return result.
        return result
    
    def find_macro(self, name: str) -> AssemblyMacro | None:
        # Check local macros.
        if name in self.macros:
            return self.macros[name]
            
        # Check parents.
        next_parent = self.parent
        while next_parent is not None:
            if name in next_parent.macros:
                return next_parent.macros[name]
            next_parent = next_parent.parent

        # Nothing found.
        return None 

@dataclass
class AssemblyLabel():
    name: str
    location: int
    scope: AssemblyScope
    source: AssemblySourceLine | None

@dataclass
class AssemblyMacro():
    name: str
    value: str
    scope: AssemblyScope
    source: AssemblySourceLine | None

@dataclass
class AssemblySourceLine():
    unit: str
    line: int
    text: str

@dataclass
class AssemblyInstruction():
    source: AssemblySourceLine
    text: str
    scope: AssemblyScope

@dataclass
class AssembledProgram():
    binary: npt.NDArray[np.uint64]
    text: list[str]
    instructions: list[AssemblyInstruction]
    global_scope: AssemblyScope

    @property
    def n_instructions(self) -> int:
        return len(self.instructions)

# region assembler exceptions

class AssemblyError(Exception):
    source_line: AssemblySourceLine
    message: str

    def __init__(self, source_line: AssemblySourceLine, message: str):
        super().__init__(f"Failed to parse line {source_line.line + 1} of unit '{source_line.unit}': '{source_line.text}'.\n{message}")
        self.source_line = source_line
        self.message = message

class AssemblySyntaxError(Exception):
    source_line: AssemblySourceLine
    message: str

    def __init__(self, source_line: AssemblySourceLine, message: str):
        super().__init__(f"Failed to parse line {source_line.line + 1} of unit '{source_line.unit}': '{source_line.text}'.\n{message}")
        self.source_line = source_line
        self.message = message

class AssemblyIncludeError(Exception):
    source_line: AssemblySourceLine
    target: str

    def __init__(self, source_line: AssemblySourceLine, target: str):
        super().__init__(f"Failed to resolve included unit '{target}' in line {source_line.line + 1} of unit '{source_line.unit}'.")
        self.source_line = source_line
        self.target = target

# endregion assembler exceptions

# region instruction parser

def parse_register(text: str) -> Register | None:  
    text = text.upper()
    try:
        register = Register[text]
        return register
    except KeyError:
        return None
    
def is_register(text: str) -> bool:
    text = text.upper()
    try:
        _ = Register[text]
        return True
    except KeyError:
        return False
    
def parse_value(text: str) -> Register | np.uint16 | None:
    # Check if value is a register.
    register = parse_register(text)
    if register is not None:
        return register
    
    # Check if value is an immediate value.
    immediate_value = parse_immediate_value(text)
    if immediate_value is not None:
        return immediate_value
    
    # Invalid value
    return None

def is_value(text: str) -> bool:
    # Check if value is a register.
    register = parse_register(text)
    if register is not None:
        return True
    
    # Check if value is an immediate value.
    immediate_value = parse_immediate_value(text)
    if immediate_value is not None:
        return True
    
    # Invalid value
    return False

def parse_immediate_value(text: str) -> np.uint16 | None:
    try:
        return np.uint16(int(text, 0) & 0xFFFF)
    except ValueError:
        return None

def is_immediate_value(text: str) -> bool:
    try:
        _ = int(text, 0)
        return True
    except ValueError:
        return False
    
condition_operators = {
    "=": Condition.EQUAL,
    "!=": Condition.NOT_EQUAL,
    ">": Condition.GREATER,
    "<": Condition.LESS,
    ">=": Condition.GREATER_OR_EQUAL,
    "<=": Condition.LESS_OR_EQUAL,
}

# Binary operators.
# These are operations with the following syntax: <output> = <a> <operator> <b>
binary_operations = {
    'and': Operation.AND,
    'nand': Operation.NAND,
    'or': Operation.OR,
    'nor': Operation.NOR,
    'xor': Operation.XOR,
    'xnor': Operation.XNOR,
    '+': Operation.ADD,
    '-': Operation.SUB,
    '*': Operation.MULTIPLY,
    '/': Operation.DIVIDE,
    '%': Operation.MODULO,
}

# Unary operators.
# These are operations with the following syntax: <output> = <operator> <input>
# Internally the input is send to A as well as B.
unary_operators = {
    'not': Operation.NAND,
    'inc': Operation.INC,
    'dec': Operation.DEC,
    'sqrt': Operation.SQRT,
}

# Instructions that take no arguments. Internally the register R1 is used as A, B and output
no_args_instructions = {
    'nop': Operation.AND,
    'break': Operation.BREAK,
}

def _parse_instruction(instruction: AssemblyInstruction) -> np.uint64:
    source_line = instruction.source
    instruction_text = instruction.text
    instruction_parts = instruction_text.split(" ")
    n_instruction_parts = len(instruction_parts)

    # Handle conditions
    if "if" in instruction_parts:
        if_index = instruction_parts.index("if")
        condition_parts = instruction_parts[(if_index + 1):]
        n_condition_parts = len(condition_parts)
        instruction_parts = instruction_parts[:if_index]
        n_instruction_parts = len(instruction_parts)

        # Check general condition syntax.
        if n_condition_parts < 3 or n_condition_parts > 4 or condition_parts[2] != "0":
            raise AssemblySyntaxError(source_line, "Invalid condition syntax. Should be 'if <register> <condition> 0'.")

        # Check if not is present.
        if n_condition_parts == 4 and condition_parts[3] != "not":
            raise AssemblySyntaxError(source_line, "Invalid condition syntax. Should be 'if <register> <condition> 0 [not]'.")

        # Get condition.
        if condition_parts[1] not in condition_operators:
            raise AssemblySyntaxError(source_line, f"Invalid condition operator '{condition_parts[1]}'.")
        condition = condition_operators[condition_parts[1]]

        # Get condition source register.
        condition_register = parse_register(condition_parts[0])
        if condition_register is None:
            raise AssemblySyntaxError(source_line, f"Invalid condition source register '{condition_parts[0]}'.")

        # If "not" is specified, invert the condition.
        if n_condition_parts == 4:
            condition = inverse_condition(condition)

    else:
        condition_register = Register.R14
        condition = Condition.ALWAYS

    # RETURN
    if instruction_text == "return":
        return encode_instruction(Operation.STACK_POP, condition_register, condition, Register.PC, 0, 0)

    # NO ARGS
    if instruction_text in no_args_instructions:
        operation = no_args_instructions[instruction_text]
        return encode_instruction(operation, condition_register, Condition.NEVER, Register.R1, Register.R1, Register.R1)

    # JUMP
    if instruction_parts[0] == 'jump':
        # Check general syntax.
        if n_instruction_parts != 2:
            raise AssemblySyntaxError(source_line, "Invalid syntax for jump. Should be 'jump <register|immediate>'.")

        # Parse the jump target value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid jump target '{instruction_parts[1]}'.")
        
        # Encode instruction.
        return encode_instruction(Operation.A, condition_register, condition, Register.PC, value, 0)
        
    # SKIP
    if instruction_parts[0] == 'skip':
        # Check general syntax.
        if n_instruction_parts != 2:
            raise AssemblySyntaxError(source_line, "Invalid syntax for skip. Should be 'skip <register|immediate>'.")

        # Parse the skip value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid skip length '{instruction_parts[1]}'.")
        
        # Encode instruction.
        return encode_instruction(Operation.ADD, condition_register, condition, Register.PC, value, Register.PC)

    # Assignments, that are instructions of the form '<output_register> = ...'
    if n_instruction_parts >= 2 and is_register(instruction_parts[0]) and instruction_parts[1] == "=":
        # Get the output register
        output_register = parse_register(instruction_parts[0])
        if output_register is None:
            raise AssemblySyntaxError(source_line, f"Invalid register '{instruction_parts[0]}'.")

        # LOAD
        if n_instruction_parts == 3:
            value_text = instruction_parts[2]

            # value, '<r> = <value>'
            value = parse_value(value_text)
            if value is not None:
                return encode_instruction(Operation.A, condition_register, condition, output_register, value, Register.R1)

            # memory, '<r> = [<address>]'
            if value_text[0] == "[" and value_text[-1] == "]":
                # Parse the address
                address_text = value_text[1:-1]
                address = parse_value(address_text)
                if address is None:
                    raise AssemblySyntaxError(source_line, f"Invalid value for memory address: '{address_text}'.")

                # Encode instruction.
                return encode_instruction(Operation.MEMORY_LOAD, condition_register, condition, output_register, address, 0)

            # io poll, '<r> = poll'
            if value_text == "poll":
                return encode_instruction(Operation.IO_POLL, condition_register, condition, output_register, 0, 0)

            # io read, '<r> = read'
            if value_text == "read":
                return encode_instruction(Operation.IO_READ, condition_register, condition, output_register, 0, 0)
            
            # stack peek, '<r> = peek'
            if value_text == "peek":
                return encode_instruction(Operation.STACK_PEEK, condition_register, condition, output_register, 0, 0)
            
            # stack pop, '<r> = pop'
            if value_text == "pop":
                return encode_instruction(Operation.STACK_POP, condition_register, condition, output_register, 0, 0)

            # Unsupported load value.
            raise AssemblySyntaxError(source_line, f"Invalid load value: '{value_text}'.")

        # SHIFT
        # TODO: shift count?
        if n_instruction_parts == 5 and instruction_parts[2] == "shift":
            # Parse value.
            value = parse_value(instruction_parts[4])
            if value is None:
                raise AssemblySyntaxError(source_line, f"Invalid value '{instruction_parts[4]}'.")

            # Parse direction.
            match instruction_parts[3]:
                case "left":
                    return encode_instruction(Operation.SHIFT_LEFT, condition_register, condition, output_register, value, 1)
                case "right":
                    return encode_instruction(Operation.SHIFT_RIGHT, condition_register, condition, output_register, value, 1)
                case _:
                    raise AssemblySyntaxError(source_line, f"Invalid shift direction '{instruction_parts[3]}'.")
        
        # ROTATE
        if n_instruction_parts == 5 and instruction_parts[2] == "rotate":
            # Parse value.
            value = parse_value(instruction_parts[4])
            if value is None:
                raise AssemblySyntaxError(source_line, f"Invalid value '{instruction_parts[4]}'.")

            # Parse direction.
            match instruction_parts[3]:
                case "left":
                    return encode_instruction(Operation.ROTATE_LEFT, condition_register, condition, output_register, value, 1)
                case "right":
                    return encode_instruction(Operation.ROTATE_RIGHT, condition_register, condition, output_register, value, 1)
                case _:
                    raise AssemblySyntaxError(source_line, f"Invalid rotation direction '{instruction_parts[3]}'.")

        # Bit operation, '<r> = <value> bit <operation> <bit>'
        if n_instruction_parts == 6 and instruction_parts[3] == "bit":
            # Parse value.
            value = parse_value(instruction_parts[2])
            if value is None:
                raise AssemblySyntaxError(source_line, f"Invalid value '{instruction_parts[2]}'.")
            
            # Parse bit.
            bit = parse_value(instruction_parts[5])
            if bit is None:
                raise AssemblySyntaxError(source_line, f"Invalid bit '{instruction_parts[5]}'.")
            
            # Parse bit operation.
            bit_operation = instruction_parts[4]
            match bit_operation:
                case "get":
                    return encode_instruction(Operation.BIT_GET, condition_register, condition, output_register, value, bit)
                case "set":
                    return encode_instruction(Operation.BIT_SET, condition_register, condition, output_register, value, bit)
                case "clear":
                    return encode_instruction(Operation.BIT_CLEAR, condition_register, condition, output_register, value, bit)
                case "toggle":
                    return encode_instruction(Operation.BIT_TOGGLE, condition_register, condition, output_register, value, bit)
                case _:
                    raise AssemblySyntaxError(source_line, f"Invalid bit operation '{bit_operation}'.")

        # Unary operators
        if n_instruction_parts == 4 and instruction_parts[2] in unary_operators:
            # Parse value.
            value = parse_value(instruction_parts[3])
            if value is None:
                raise AssemblySyntaxError(source_line, f"Invalid value '{instruction_parts[3]}'.")
            
            # Parse operation.
            operation = unary_operators[instruction_parts[2]]

            # Encode instruction.
            return encode_instruction(operation, condition_register, condition, output_register, value, 0)

        # Binary operators
        if n_instruction_parts == 5 and instruction_parts[3] in binary_operations:
            # Parse value for A.
            value_a = parse_value(instruction_parts[2])
            if value_a is None:
                raise AssemblySyntaxError(source_line, f"Invalid value for A: '{instruction_parts[3]}'.")
            
            # Parse value for B.
            value_b = parse_value(instruction_parts[4])
            if value_b is None:
                raise AssemblySyntaxError(source_line, f"Invalid value for A: '{instruction_parts[3]}'.")

            # Parse operation.
            operation = binary_operations[instruction_parts[3]]

            # Encode instruction.
            return encode_instruction(operation, condition_register, condition, output_register, value_a, value_b)

        raise AssemblySyntaxError(source_line, "Invalid right hand side for assignment")

    # Memory store, '[<address>] = <value>'
    if n_instruction_parts == 3 and instruction_parts[0][0] == "[" and instruction_parts[0][-1] == "]" and instruction_parts[1] == "=":
        # Parse address.
        address_text = instruction_parts[0][1:-1]
        address = parse_value(address_text)
        if address is None:
            raise AssemblySyntaxError(source_line, f"Invalid memory address: '{address_text}'.")
        
        # Parse value.
        value = parse_value(instruction_parts[2])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid value: '{instruction_parts[2]}'.")
    
        # Encode instruction.
        return encode_instruction(Operation.MEMORY_STORE, condition_register, Condition.NEVER, Register.R1, address, value)

    # Stack push, 'push <value>'
    if n_instruction_parts == 2 and instruction_parts[0] == "push":
        # Parse value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid value: '{instruction_parts[1]}'.")

        # Encode instruction.
        return encode_instruction(Operation.STACK_PUSH, condition_register, Condition.NEVER, Register.R1, value, 0)

    # Call, 'call <address>'
    if n_instruction_parts == 2 and instruction_parts[0] == "call":
        # Parse value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid value: '{instruction_parts[1]}'.")

        # Encode instruction.
        return encode_instruction(Operation.STACK_CALL, condition_register, condition, Register.PC, Register.PC, value)

    # IO WRITE, 'write <value>'
    if n_instruction_parts == 2 and instruction_parts[0] == "write":
        # Parse value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid value: '{instruction_parts[1]}'.")

        # Encode instruction.
        return encode_instruction(Operation.IO_WRITE, condition_register, Condition.NEVER, Register.R1, value, 0)
    
    # No instruction pattern detected, throw generic syntax exception.
    raise AssemblySyntaxError(source_line, f"Invalid instruction: '{instruction_text}'.")

# endregion instruction parser

# region assembler

def _preformat_source_line(line: str) -> str:
    # Remove comments
    line = line.split('#', 1)[0]

    # Strip line
    line = line.strip()

    # Simplify whitespace.
    line = line.replace("    ", " ")
    line = " ".join(line.split())

    # Convert to lowercase.
    line = line.lower()

    return line

def _prepare_instructions(
    src_lines: list[str],
    unit: str,
    include_directories: list[pathlib.Path],
    global_scope: AssemblyScope,
    included_units: set[str] | None = None,
) -> list[AssemblyInstruction]:
    # Early exit if program is empty.
    if len(src_lines) == 0:
        return []
    
    # Prepare include scope.
    if included_units is None:
        included_units = set()
    included_units = included_units | { unit }
    
    # Create the initial source mapping.
    instructions: list[AssemblyInstruction] = [ AssemblyInstruction(AssemblySourceLine(unit, line, text.strip()), _preformat_source_line(text), global_scope) for (line, text) in enumerate(src_lines) ]

    # Remove empty lines.
    i_line = 0
    while i_line < len(instructions):
        if instructions[i_line].text == "":
            # Delete line, DON'T increment line counter.
            del instructions[i_line]
        else:
            # Increment line counter.
            i_line += 1

    # Handle includes.
    i_line = 0
    while i_line < len(instructions):
        line_text = instructions[i_line].text
        if not line_text.startswith("!include "):
            i_line += 1
            continue

        # Parse include parts.
        include_parts = line_text.split(" ")
        n_include_parts = len(include_parts)
        if n_include_parts < 2:
            raise AssemblySyntaxError(instructions[i_line].source, "Invalid include statement, missing input argument")
        
        # Include statement.
        include_target_id = include_parts[1]
        include_file_path: pathlib.Path | None = None
        for include_directory in include_directories:
            file_path = include_directory / f"{include_target_id}.mcasm"
            if file_path.exists() and file_path.is_file():
                include_file_path = file_path
                break

        # Raise exception if include target cannot be resolved.
        if include_file_path is None:
            raise AssemblyIncludeError(instructions[i_line].source, include_target_id)
        
        # Read include file
        try:
            with open(include_file_path, "r") as include_file:
                include_src_lines = include_file.readlines()
        except Exception:
            raise AssemblyIncludeError(instructions[i_line].source, include_target_id) 

        # Prepare include lines recursivly.
        include_instructions = _prepare_instructions(include_src_lines, str(include_file_path), include_directories, global_scope, included_units)

        # Check if a include scope is specified.
        # In this case all instructions of `include_instructions` outside of the target scope are filtered out,
        # so that only the target scope is included. 
        if n_include_parts > 2:
            # Parse target scope name.
            include_scope_name = include_parts[2]
            if not include_scope_name.startswith("@"):
                raise AssemblySyntaxError(instructions[i_line].source, f"Invalid scope name '{include_scope_name}' doesn't start with '@'.")

            # Search for scope start. ('<label> {').
            while len(include_instructions) > 0:
                if include_instructions[0].text == f"{include_scope_name} {'{'}":
                    break
                del include_instructions[0]
            else:
                raise AssemblyError(instructions[i_line].source, f"No scope with the name '{include_scope_name}' found in unit '{str(include_file_path)}'.")

            # Search for corresponding closing bracket.
            include_scope_length = 1
            i_scope = 1
            while include_scope_length < len(include_instructions):

                # Scope is closed.
                if "}" in include_instructions[include_scope_length].text:
                    i_scope -= 1

                    # Check if main scope was closed.
                    if i_scope <= 0:
                        # Add one to length and return.
                        include_scope_length += 1
                        break

                # Open sub scope.
                elif "{" in include_instructions[include_scope_length].text:
                    i_scope += 1

                # Increment length.
                include_scope_length += 1
            else:
                raise AssemblyError(instructions[i_line].source, f"Scope '{include_scope_name}' in unit '{str(include_file_path)}' is not closed.")

            # Remove remaining include instructions.
            include_instructions = include_instructions[:include_scope_length]

            # Check if scope alias is specified
            if n_include_parts > 3:
                if n_include_parts != 5 or include_parts[3] != "as":
                    raise AssemblySyntaxError(instructions[i_line].source, "Invalid include. Syntax is '!include <unit> @<scope> as @<alias>'.")

                # Parse alias scope name.
                scope_alias_name = include_parts[4]
                if not scope_alias_name.startswith("@"):
                    raise AssemblySyntaxError(instructions[i_line].source, f"Invalid scope alias name '{scope_alias_name}' doesn't start with '@'.")

                # Replace scope name with alias.
                for instruction in include_instructions:
                    # Add whitespace to fix word replace at line start / end
                    instruction.text = f" {instruction.text} "

                    # Apply labels.
                    instruction.text = instruction.text.replace(f" {include_scope_name} ", f" {scope_alias_name} ")
                    instruction.text = instruction.text.replace(f"[{include_scope_name}]", f"[{scope_alias_name}]")

                    # Remove previously added whitespace.
                    instruction.text = instruction.text[1:-1]
                                
        # Replace include statement with included lines.
        instructions = instructions[:i_line] + include_instructions + instructions[(i_line+1):]

        # Skip included lines as they have already been processed.
        i_line += len(include_instructions)

    return instructions

def assemble(
    src_lines: list[str],
    unit: str,
    include_directories: list[pathlib.Path],
    default_macros: dict[str, str] = {},
    default_labels: dict[str, int] = {},
) -> AssembledProgram | None:
    # Define the global scope.
    global_scope = AssemblyScope(0, GLOBAL_SCOPE_NAME, None, [], 0, -1, {}, {})
    for (name, location) in default_labels.items():
        global_scope.labels[name] = AssemblyLabel(name, location, global_scope, None)
    for (name, value) in (ASSEMBLER_MACROS | default_macros).items():
        global_scope.macros[name] = AssemblyMacro(name, value, global_scope, None)

    # Early exit if program is empty.
    if len(src_lines) == 0:
        return AssembledProgram(np.zeros(1, dtype=np.uint64), [], [], global_scope)

    # Prepare the source lines.
    # This handles things like comments, whitespace simplifcation and lowercase transforamtion, 
    # as well as the !include statement.
    instructions = _prepare_instructions(src_lines, str(unit), include_directories, global_scope)
    
    # Insert initialization jump.
    instructions.insert(0, AssemblyInstruction(AssemblySourceLine(GENERATED_UNIT_NAME, 0, f"jump @{START_LABEL_NAME}"), f"jump @{START_LABEL_NAME}", global_scope))

    # Preprocessor
    # Handles labels, macros
    next_scope_id: int = 1
    current_scope = global_scope
    i_instruction = 0
    while i_instruction < len(instructions):
        instruction = instructions[i_instruction]

        # Update instruction scope.
        instruction.scope = current_scope
        
        # Handle scopes.
        if instruction.text == "{":
            # Create new child scope.
            scope = AssemblyScope(next_scope_id, f"__unnamed_scope_{next_scope_id}__", current_scope, [], i_instruction, -1, {}, {})
            current_scope.children.append(scope)
            current_scope = scope
            next_scope_id += 1

            # Delete original instruction line, DON'T increment line counter.
            del instructions[i_instruction]
            continue

        if instruction.text == "}":
            # Check if there is a parent scope.
            parent_scope = current_scope.parent
            if parent_scope is None:
                raise AssemblyError(instruction.source, "Can't close the global scope.")
            
            # Close scope.
            current_scope.end = i_instruction
            current_scope = parent_scope

            # Delete original instruction line, DON'T increment line counter.
            del instructions[i_instruction]
            continue

        # Handle labels.
        if instruction.text.startswith("@"):
            # Parse label.
            label_parts = instruction.text[1:].split(" ", maxsplit=1)
            n_label_parts = len(label_parts)
            create_scope = False
            match n_label_parts:
                case 0:
                    raise AssemblySyntaxError(instruction.source, f"Invalid label '{instruction.text}'.")
                case 1:
                    label_name = label_parts[0]
                case 2:
                    label_name = label_parts[0]
                    if label_parts[1] != "{":
                        raise AssemblySyntaxError(instruction.source, f"Invalid label '{instruction.text}'.")
                    create_scope = True
                case _:
                    raise AssemblySyntaxError(instruction.source, f"Invalid label '{instruction.text}'.")
                
            # Create label.
            label = current_scope.create_label(label_name, i_instruction, instruction.source)

            # Create scope if specified.
            if create_scope:
                # Create new child scope.
                scope = AssemblyScope(next_scope_id, label.name, current_scope, [], i_instruction, -1, {}, {})
                current_scope.children.append(scope)
                current_scope = scope
                next_scope_id += 1

            # Delete original instruction line, DON'T increment line counter.
            del instructions[i_instruction]
            continue

        # Handle macros.
        if instruction.text.startswith("!define"):
            # Parse macro.
            define_parts = instruction.text.split(" ", 2)
            if len(define_parts) < 3:
                raise AssemblySyntaxError(instruction.source, "Invalid macro definition. Syntax is '!define <name> <value>'.")
            macro_name = define_parts[1]
            macro_value = " ".join(define_parts[2:])

            # Create macro.
            current_scope.create_macro(macro_name, macro_value, instruction.source)

            # Delete original instruction line, DON'T increment line counter.
            del instructions[i_instruction]
            continue

        # Increment line counter.
        i_instruction += 1

    # Make sure that the start label is defined.
    if global_scope.find_label(START_LABEL_NAME) is None:
        # If the start label has not been defined in the source code, define it here at instruction 1.
        # Instruction 0 must be skipped, as this is the initial jump.
        global_scope.create_label(START_LABEL_NAME, 1, None)

    # Count instructions. The number of instruction doesn't change after this point.
    n_instructions = len(instructions)

    # Transform repeat / exit statements into jumps.
    for instruction in instructions:
        # Check if instruction is a repeat.
        instruction_parts = instruction.text.split(" ")
        n_instruction_parts = len(instruction_parts)
        if n_instruction_parts < 1 or (instruction_parts[0] not in ["repeat", "exit"]):
            continue

        i_remaining_parts = 1

        # Check if label is specified
        scope = instruction.scope
        if n_instruction_parts > 1 and instruction_parts[1].startswith("@"):
            target_scope_name = instruction_parts[1][1:] # Skip '@' symbol.
            i_remaining_parts = 2

            # Search scope with name.
            while scope is not None:
                if scope.name == target_scope_name:
                    break
                scope = scope.parent
            else:
                # This is only executed if the loop exited "naturally", which means the scope was not found.
                raise AssemblySyntaxError(instruction.source, f"Unknown parent scope '@{target_scope_name}'")
        
        remaining_part = " ".join(instruction_parts[i_remaining_parts:])

        match instruction_parts[0]:
            case "repeat":
                instruction.text = f"jump {scope.start} {remaining_part}"
            case "exit":
                instruction.text = f"jump {scope.end} {remaining_part}"
            case _:
                raise AssemblyError(instruction.source, f"Invalid scope statement '{instruction_parts[0]}'")

    # Apply labels and macros.
    for instruction in instructions:
        # Add whitespace to fix word replace at line start / end
        instruction.text = f" {instruction.text} "

        # Apply macros.
        # Because the value of a macro could be the key of another macro, all macros are applied repeatedly,
        # until there are no more changes.
        # To check for cyclic dependencies, previous_instruction_states keeps track of 
        # all previously seen states of this line.
        previous_instruction_states: set[str] = set()
        while instruction.text not in previous_instruction_states:
            previous_instruction_states.add(instruction.text)

            # Apply all macros and count the numbe
            unchanged = True
            for macro in instruction.scope.visible_macros().values():
                if f" {macro.name} " in instruction.text:
                    instruction.text = instruction.text.replace(f" {macro.name} ", f" {macro.value} ")
                    unchanged = False
                if f"[{macro.name}]" in instruction.text:
                    instruction.text = instruction.text.replace(f"[{macro.name}]", f"[{macro.value}]")
                    unchanged = False

            # Break if line remained unchanged.
            if unchanged:
                break
        else:
            raise AssemblyError(instruction.source, "Cyclic macro definition.")

        # Check for unresolved macros.
        if " $" in instruction.text:
            i_macro_start = instruction.text.index(" $") + 1
            if " " in instruction.text[i_macro_start:]:
                i_macro_end = instruction.text[i_macro_start:].index(" ")
                macro_id = instruction.text[i_macro_start:(i_macro_end + i_macro_start)]
            else:
                macro_id = instruction.text[i_macro_start]
            raise AssemblyError(instruction.source, f"Unable to resolve macro '{macro_id}'.")
        
        if "[$" in instruction.text:
            i_macro_start = instruction.text.index("[$") + 1
            if "]" in instruction.text[i_macro_start:]:
                i_macro_end = instruction.text[i_macro_start:].index(" ")
                macro_id = instruction.text[i_macro_start:(i_macro_end + i_macro_start)]
            else:
                macro_id = instruction.text[i_macro_start]
            raise AssemblyError(instruction.source, f"Unable to resolve macro '{macro_id}'.")

        # Apply labels.
        for label in instruction.scope.visible_labels().values():
            instruction.text = instruction.text.replace(f" @{label.name} ", f" {label.location} ")
            instruction.text = instruction.text.replace(f"[@{label.name}]", f"[{label.location}]")

        # Check for unresolved labels.
        if " @" in instruction.text:
            i_label_start = instruction.text.index(" @") + 1
            if " " in instruction.text[i_label_start:]:
                i_label_end = instruction.text[i_label_start:].index(" ")
                label_id = instruction.text[i_label_start:(i_label_end + i_label_start)]
            else:
                label_id = instruction.text[i_label_start]
            raise AssemblyError(instruction.source, f"Unable to resolve label '{label_id}'.")

        if "[@" in instruction.text:
            i_label_start = instruction.text.index("[@") + 1
            if " " in instruction.text[i_label_start:]:
                i_label_end = instruction.text[i_label_start:].index(" ")
                label_id = instruction.text[i_label_start:(i_label_end + i_label_start)]
            else:
                label_id = instruction.text[i_label_start]
            raise AssemblyError(instruction.source, f"Unable to resolve label '{label_id}'.")
            
        # Remove previously added whitespace.
        instruction.text = instruction.text[1:-1]

    # Convert every standalone if condition that is followed by a new scope to a condition skip, that skips that scope.
    # Note that the condition must be inverted for that.
    for (i_instruction, instruction) in enumerate(instructions):
        instruction = instructions[i_instruction]
        # Check if instruction is a standalone if condition.
        if not instruction.text.startswith("if"):
            continue

        # Check that condition has a target scope.
        if i_instruction == n_instructions - 1:
            raise AssemblySyntaxError(instruction.source, "Condition without target scope")

        next_instruction = instructions[i_instruction + 1]
        if next_instruction.scope.id == instruction.scope.id:
            # Next instrution is in the same scope, skip only one instruction (+1 to skip the condition itself).
            # The condition is inverted by appending a "not".
            instruction.text = f"skip 2 {instruction.text} not"
            continue

        # Find the scope that should be skipped.
        # This would be the scope, whoes parent scope is the scope of the if.
        scope = next_instruction.scope
        while scope is not None:
            if scope.parent is not None and scope.parent.id == instruction.scope.id:
                break
            scope = scope.parent
        else:
            # This is only executed if the loop exited "naturally", which means the scope was not found.
            raise AssemblySyntaxError(instruction.source, "Standalone condition must be placed before a scope")
        
        # Skip the length of the following scope (+1 to skip the condition iftself).
        # The condition is inverted by appending a "not".
        instruction.text = f"skip {scope.end - scope.start + 1} {instruction.text} not"
        continue

    # Assemble instructions.
    encoded_instructions = np.zeros(n_instructions, dtype=np.uint64)
    for i_instruction, instruction in enumerate(instructions):
        try:
            # Parse the instruction.
            encoded_instructions[i_instruction] = _parse_instruction(instruction)
        except AssemblySyntaxError:
            raise
        except Exception as exception:
            raise Exception(f"Exception whilst parsing line {instruction.source.line + 1} of unit '{instruction.source.unit}': '{instruction.source.text}'.") from exception

    # Return generated program.
    return AssembledProgram(encoded_instructions, src_lines, instructions, global_scope)

# endregion assembler

# region application

def program_to_memory(instructions: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    n_instructions = len(instructions)
    memory = np.zeros(PROGRAM_MEMORY_SIZE, dtype=np.uint64)
    memory[0:n_instructions] = instructions[0:n_instructions]
    return memory

# MAIN PROGRAM
if __name__ == "__main__":
    # Parse arguments
    argument_parser = argparse.ArgumentParser(
        prog="MCASM",
        description="Assembler for the MCPC",
    )
    
    argument_parser.add_argument("filename", nargs="?", default="./programs/stdlib_test.mcasm")
    argument_parser.add_argument("-o", "--output")
    argument_parser.add_argument("-c", "--check", action="store_true")
    argument_parser.add_argument("-p", "--pre", action="store_true")
    arguments = argument_parser.parse_args()
    input_filename: str = arguments.filename
    output_filename: str | None = arguments.output
    check_mode: bool = arguments.check
    output_preprocessed: bool = arguments.pre

    # Resolve input and output file path.
    input_filepath = pathlib.Path(input_filename)
    if output_filename is None:
        output_filename = f"output/{input_filepath.stem}.mcbin"
    output_filepath = pathlib.Path(output_filename)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Prepare include paths.
    include_paths: list[pathlib.Path] = [
        pathlib.Path.cwd() / "stdlib",
    ]
    include_paths = [ path for path in include_paths if path.exists() and path.is_dir() ]

    # Read input lines
    with open(input_filepath, "r") as input_file:
        src_lines = input_file.readlines()

    # Assemble the program
    program = None
    try:
        program = assemble(src_lines, str(input_filepath.absolute()), include_paths)
    except AssemblyError as exception:
        print(exception, file=sys.stderr)
        exit(1)
    except AssemblySyntaxError as exception:
        print(exception, file=sys.stderr)
        exit(1)
    except AssemblyIncludeError as exception:
        print(exception, file=sys.stderr)
        exit(1)

    if program is None:
        print("Failed to assemble program.", file=sys.stderr)
        exit(1)

    # Write the output to a file.
    if not check_mode:
        with open(output_filepath, "wb") as output_file:
            output_file.write(program.binary.tobytes())

    # Output preprocessed assembly if flag is enabled. 
    if output_preprocessed:
        mappings_path = output_filepath.parent / f"{output_filepath.stem}.pre.mcasm"
        with open(mappings_path, "w") as mappings_file:
            max_instruction_line_length = np.max([len(line.text) for line in program.instructions])

            for instruction in program.instructions:
                if instruction.source.unit == GENERATED_UNIT_NAME:
                    mappings_file.write(f"{instruction.text.ljust(max_instruction_line_length + 3)} # generated\n")
                else:
                    mappings_file.write(f"{instruction.text.ljust(max_instruction_line_length + 3)} # line {instruction.source.line + 1:5d} of unit \"{instruction.source.unit}\"\n")

    # Summary output.
    print(f"Assembled {len(program.instructions)} instructions")

# endregion application
