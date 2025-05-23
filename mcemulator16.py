import argparse
import sys
import warnings
import numpy as np
import numpy.typing as npt
import pathlib
import time

from mcpc16 import MEMORY_SIZE, PROGRAM_MEMORY_SIZE, REGISTER_COUNT, STACK_SIZE, Condition, Instruction, Operation, Register, decode_program
import mcasm16

class Emulator:
    program: list[Instruction] = []
    registers: npt.NDArray[np.uint16] = np.zeros(REGISTER_COUNT, dtype=np.uint16)
    memory: npt.NDArray[np.uint16] = np.zeros(MEMORY_SIZE, dtype=np.uint16)
    stack: npt.NDArray[np.uint16] = np.zeros(STACK_SIZE, dtype=np.uint16)
    stack_pointer = np.uint16(0xFFFF)
    halt: bool = False
    carry: bool = False

    @property
    def pc(self) -> np.uint16:
        return self.registers[0]
    
    def register_value(self, register: Register) -> np.uint16:
        return self.registers[register.value]
    
    def set_register_value(self, register: Register, value: np.uint16):
        self.registers[register.value] = value
    
    @pc.setter
    def pc(self, value: np.uint16):
        self.registers[0] = value

    def initialize(self):
        self.registers = np.zeros(REGISTER_COUNT, dtype=np.uint16)
        self.memory = np.zeros(MEMORY_SIZE, dtype=np.uint16)
        self.stack = np.zeros(STACK_SIZE, dtype=np.uint16)
        self.stack_pointer = np.uint16(0xFFFF)

    def load_program(self, program: list[Instruction]):
        if (len(program) > PROGRAM_MEMORY_SIZE):
            raise Exception("The specified program is to big for this architecture.")
        self.program = program
        self.initialize()

    def evaluate_condition(self, condition: Condition, value: np.uint16) -> bool:
        is_zero = (value == 0)
        is_negative = ((value & 0b1000_0000_0000_0000) != 0)

        match condition:
            case Condition.ALWAYS:
                return True
            case Condition.NEVER:
                return False
            case Condition.EQUAL:
                return is_zero
            case Condition.NOT_EQUAL:
                return not is_zero
            case Condition.LESS:
                return is_negative
            case Condition.GREATER_OR_EQUAL:
                return not is_negative
            case Condition.GREATER:
                return (not is_zero) and (not is_negative)
            case Condition.LESS_OR_EQUAL:
                return is_zero or is_negative

    def evaluate_operation(self, operation: Operation, a: np.uint16, b: np.uint16) -> tuple[np.uint16, bool]:
        match operation:
            case Operation.CLEAR:
                return (np.uint16(0), True)
            case Operation.A:
                return (a, True)
            case Operation.AND:
                return (a & b, True)
            case Operation.NAND:
                return (~(a & b), True)
            case Operation.OR:
                return (a | b, True)
            case Operation.NOR:
                return (~(a | b), True)
            case Operation.XOR:
                return (a ^ b, True)
            case Operation.XNOR:
                return (~(a ^ b), True)
            
            case Operation.INC:
                return (a + 1, True)
            case Operation.DEC:
                return (a - 1, True)
            case Operation.ADD:
                a_big = np.uint32(a)
                b_big = np.uint32(b)
                self.carry = (a + b >> 16) != 0
                return (np.uint16((a_big + b_big) & 0xFFFF), True)
            case Operation.SUB:
                a_big = np.uint32(a)
                b_big = np.uint32(b)
                self.carry = (a - b >> 16) != 0
                return (np.uint16((a_big - b_big) & 0xFFFF), True)
            
            case Operation.ADD_WITH_CARRY:
                a_big = np.uint32(a)
                b_big = np.uint32(b)
                result = np.uint16((a_big + b_big + self.carry) & 0xFFFF)
                self.carry = (a + b + self.carry >> 16) != 0
                return (result, True)
            case Operation.SUB_WITH_CARRY:
                a_big = np.uint32(a)
                b_big = np.uint32(b)
                result = np.uint16((a_big - b_big - (1 - self.carry)) & 0xFFFF)
                self.carry = (a - b - (1 - self.carry) >> 16) != 0
                return (result, True)
            
            case Operation.SHIFT_LEFT:
                self.carry = (a & 0x8000) != 0
                return (np.uint16(a << 1), True)
            case Operation.SHIFT_RIGHT:
                self.carry = (a & 0x0001) != 0
                return (np.uint16(a >> 1), True)
            case Operation.ROTATE_LEFT:
                result = (np.uint16((a << 1) | (1 if self.carry else 0)), True)
                self.carry = (a & 0x8000) != 0
                return result
            case Operation.ROTATE_RIGHT:
                self.carry = (a & 0x0001) != 0
                return (np.uint16((a >> 1) | (1 if self.carry else 0)), True)
            
            case Operation.BIT_GET:
                return (np.uint16((a >> b) & 0b1 != 0), True)
            case Operation.BIT_SET:
                return (a | (np.uint16(1) << b), True)
            case Operation.BIT_CLEAR:
                return (a & ~(np.uint16(1) << b), True)
            case Operation.BIT_TOGGLE:
                return (a ^ (np.uint16(1) << b), True)

            case Operation.MEMORY_LOAD:
                return (self.memory[a], True)
            case Operation.MEMORY_STORE:
                self.memory[a] = b
                return (b, False) # Return the value
            
            case Operation.IO_POLL:
                return (np.uint16(1), True) # TODO: Actually check if input is available
            case Operation.IO_READ:
                while True:
                    value = input("[INPUT] ")
                    try:
                        return (np.uint16(int(value, 0) & 0xFFFF), True)
                    except Exception:
                        print(f"Invalid number input '{value}', please try again")
                
            case Operation.IO_WRITE:
                print(f"[OUTPUT] uint16: {a} | int16: {a.view(np.int16)}")
                return (a, False) # Return the value
            
            case Operation.STACK_PEEK:
                return (self.stack[self.stack_pointer], True)
            case Operation.STACK_CALL:
                self.stack_pointer += 1
                self.stack[self.stack_pointer & 0x1F] = self.pc + 1
                return (b, True)
            case Operation.STACK_PUSH:
                self.stack_pointer += 1
                self.stack[self.stack_pointer & 0x1F] = a
                return (b, True)
            case Operation.STACK_POP:
                value = self.stack[self.stack_pointer & 0x1F]
                self.stack_pointer -= 1
                return (value, True)
            
            case Operation.MULTIPLY:
                return (a * b, True)
            case Operation.DIVIDE:
                return (a // b, True)
            case Operation.MODULO:
                return (a % b, True)
            case Operation.SQRT:
                return (np.uint16(int(np.sqrt(a))), True)
            
            case Operation.BREAK:
                self.halt = True
                return (a, False)
            
            case _:
                raise Exception(f"Operation {operation} ({operation.value}) is not implemented")

    def execute_instruction(self):
        # Fetch instruction.
        if self.pc >= 0 and self.pc < len(self.program):
            instruction = self.program[self.pc]
        else:
            instruction = Instruction(Operation.A, Register.R15, Condition.ALWAYS, Register.PC, np.uint(0), np.uint(0))

        # Execute instruction.
        a_value = self.register_value(instruction.a) if isinstance(instruction.a, Register) else instruction.a
        b_value = self.register_value(instruction.b) if isinstance(instruction.b, Register) else instruction.b

        condition_value = self.register_value(instruction.condition_register)
        condition_valid = self.evaluate_condition(instruction.condition, condition_value)

        operation_result, operation_has_result = self.evaluate_operation(instruction.operation, a_value, b_value)

        # Increment program counter.
        self.pc += 1

        # Store result.
        if condition_valid & operation_has_result:
            self.set_register_value(instruction.output_register, operation_result)

    def execute_instructions(self):
        while not self.halt:
            self.execute_instruction()
        self.halt = False

# MAIN PROGRAM
if __name__ == "__main__":
    # Parse arguments
    argument_parser = argparse.ArgumentParser(
        prog="MCEMULATOR",
        description="Emulator for the MCPC",
    )
    argument_parser.add_argument("filename", nargs="?", default="./programs/calculator.mcasm")
    argument_parser.add_argument("-t", "--time")
    arguments = argument_parser.parse_args()
    input_filename: str = arguments.filename
    clock_time: float = float(arguments.time or 0.0)

    # Disable numpy warnings
    warnings.filterwarnings("ignore")

    # Read input lines
    input_filepath = pathlib.Path(input_filename)
    if input_filepath.suffix == ".mcasm":
        # Program is assembled source code.
        with open(input_filepath, "r") as input_file:
            src_lines = [line.strip() for line in input_file.readlines()]

        # Prepare include paths.
        include_paths: list[pathlib.Path] = [
            pathlib.Path.cwd() / "stdlib",
        ]
        include_paths = [ path for path in include_paths if path.exists() and path.is_dir() ]

        # Assemble the program
        assembled_program = None
        try:
            assembled_program = mcasm16.assemble(src_lines, str(input_filepath.absolute()), include_paths)
        except mcasm16.AssemblyError as exception:
            print(exception, file=sys.stderr)
            exit(1)
        except mcasm16.AssemblySyntaxError as exception:
            print(exception, file=sys.stderr)
            exit(1)
        except mcasm16.AssemblyIncludeError as exception:
            print(exception, file=sys.stderr)
            exit(1)

        if assembled_program is None:
            print("Failed to assemble program.", file=sys.stderr)
            exit(1)

        print(f"Assembled {len(assembled_program.statements)} instructions")
        program = assembled_program.instructions
    
    else:
        # Program is binary.
        with open(input_filepath, "rb") as input_file:
            encoded_program = np.frombuffer(input_file.read(), dtype=np.uint64)
            program = decode_program(encoded_program)
            print(f"Loaded {len(program)} instructions")

    emulator = Emulator()
    emulator.load_program(program)
    try:
        while True:
            emulator.execute_instruction()
            if emulator.halt:
                input("Press any key to continue...")
                emulator.halt = False
            if clock_time > 0:
                time.sleep(clock_time)
    except KeyboardInterrupt:
        exit(0)
