from dataclasses import dataclass
from enum import Enum
import numpy as np

REGISTER_COUNT = 0x10
PROGRAM_MEMORY_SIZE = 0x10000
MEMORY_SIZE = 0x10000
STACK_SIZE = 32

class Register(Enum):
    PC = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    R8 = 8
    R9 = 9
    R10 = 10
    R11 = 11
    R12 = 12
    R13 = 13
    R14 = 14
    R15 = 15

class Operation(Enum):
    CLEAR = 0
    A = 1
    AND = 2
    NAND = 3
    OR = 4
    NOR = 5
    XOR = 6
    XNOR = 7

    INC = 8
    DEC = 9
    ADD = 10
    SUB = 11

    SHIFT_LEFT = 12
    SHIFT_RIGHT = 13
    ROTATE_LEFT = 14
    ROTATE_RIGHT = 15

    MEMORY_LOAD = 16
    MEMORY_STORE = 17

    IO_POLL = 18
    IO_READ = 19
    IO_WRITE = 20

    STACK_PEEK = 21
    STACK_CALL = 22
    STACK_PUSH = 23
    STACK_POP = 24

    BIT_GET = 25
    BIT_SET = 26
    BIT_CLEAR = 27
    BIT_TOGGLE = 28

    UNDEFINED_29 = 29
    UNDEFINED_30 = 30
    UNDEFINED_31 = 31
    UNDEFINED_32 = 32
    UNDEFINED_33 = 33
    UNDEFINED_34 = 34
    UNDEFINED_35 = 35

    UNDEFINED_36 = 36
    UNDEFINED_37 = 37
    UNDEFINED_38 = 38
    UNDEFINED_39 = 39
    UNDEFINED_40 = 40
    UNDEFINED_41 = 41
    UNDEFINED_42 = 42
    UNDEFINED_43 = 43
    UNDEFINED_44 = 44
    UNDEFINED_45 = 45
    UNDEFINED_46 = 46
    UNDEFINED_47 = 47

    MULTIPLY = 48
    DIVIDE = 49
    MODULO = 50
    SQRT = 51
    UNDEFINED_52 = 52
    UNDEFINED_53 = 53
    UNDEFINED_54 = 54
    UNDEFINED_55 = 55
    UNDEFINED_56 = 56
    UNDEFINED_57 = 57
    UNDEFINED_58 = 58
    UNDEFINED_59 = 59
    UNDEFINED_60 = 60
    UNDEFINED_61 = 61
    UNDEFINED_62 = 62
    BREAK = 63

class Condition(Enum):
    ALWAYS = 0
    NEVER = 1
    EQUAL = 2
    NOT_EQUAL = 3
    LESS = 4
    GREATER_OR_EQUAL = 5
    GREATER = 6
    LESS_OR_EQUAL = 7

@dataclass
class Instruction():
    operation: Operation
    condition_register: Register
    condition: Condition
    output_register: Register
    a: Register | np.uint16
    b: Register | np.uint16

def is_valid_condition_source(register: Register) -> bool:
    match register:
        case Register.R14 | Register.R15:
            return True
        case _:
            return False

def inverse_condition(condition: Condition) -> Condition:
    return list(Condition)[condition.value ^ 1]

def condition_source(register: Register) -> int:
    match register:
        case Register.R14:
            return 0
        case Register.R15:
            return 1
        case _:
            raise Exception(f"Register {register} can't be used for conditions")

def encode_instruction(
    operation: Operation, 
    condition_register: Register, 
    condition: Condition, 
    output: Register, 
    a: Register | np.uint16 | int, 
    b: Register | np.uint16 | int,
) -> np.uint64:
    if isinstance(a, Register):
        a_value = np.uint64(a.value)
        a_mode = np.uint64(1)
    else:
        a_value = np.uint64(a)
        a_mode = np.uint64(0)

    if isinstance(b, Register):
        b_value = np.uint64(b.value)
        b_mode = np.uint64(1)
    else:
        b_value = np.uint64(b)
        b_mode = np.uint64(0)
    
    opcode = np.uint64(0)
    opcode |= np.uint64(a_mode) << 0 # A source
    opcode |= np.uint64(b_mode) << 1 # B source
    opcode |= np.uint64(output.value) << 2 # Output address
    opcode |= np.uint64(condition.value) << 6 # Condition
    opcode |= np.uint64(condition_source(condition_register) & 0b1) << 9 # Condition source
    opcode |= np.uint64(operation.value) << 10 # Operation
    opcode |= np.uint64(a_value) << 16 # A value
    opcode |= np.uint64(b_value) << 32 # B Value
    return np.uint64(opcode)

def decode_instruction(instruction: np.uint64) -> Instruction:
    # Decode a.
    a_mode = ((instruction >> 0) & 0b1) != 0
    if a_mode:
        a = list(Register)[(instruction >> 16) & 0b1111]
    else:
        a = np.uint16((instruction >> 16) & 0xFFFF)

    # Decode b.
    b_mode = ((instruction >> 1) & 0b1) != 0
    if b_mode:
        b = list(Register)[(instruction >> 32) & 0b1111]
    else:
        b = np.uint16((instruction >> 32) & 0xFFFF)

    # Decode output register.
    output_register = list(Register)[(instruction >> 2) & 0b1111]

    # Decode condition.
    condition = list(Condition)[(instruction >> 6) & 0b111]
    condition_source = list(Register)[Register.R14.value + ((instruction >> 9) & 0b1)]

    # Decode operation.
    operation = list(Operation)[((instruction >> 10) & 0b111111)]

    # Return instruction.
    return Instruction(operation, condition_source, condition, output_register, a, b)