-- Generate input mapping
local pc_inputs = {}
for i = 0, 15 do
    local block_id = "redstone_relay_" .. tostring(i + 0)
    local block = peripheral.wrap(block_id)
    if block == nil then
        printError("Failed to find input peripheral '" .. block_id .. "'")
        goto program_end
    end
    pc_inputs[i] = { block = block, side = "top" }
end

-- Generate output mapping
local instruction_outputs = {}
for i = 0, 15 do
    local block_id = "redstone_relay_" .. tostring(i + 16)
    local block = peripheral.wrap(block_id)
    if block == nil then
        printError("Failed to find instruction output peripheral '" .. block_id .. "'")
        goto program_end
    end
    instruction_outputs[i] = { block = block, side = "top" }
end
local argument_a_outputs = {}
for i = 0, 15 do
    local block_id = "redstone_relay_" .. tostring(i + 16)
    local block = peripheral.wrap(block_id)
    if block == nil then
        printError("Failed to find argument a output peripheral '" .. block_id .. "'")
        goto program_end
    end
    argument_a_outputs[i] = { block = block, side = "bottom" }
end
local argument_b_outputs = {}
for i = 0, 15 do
    local block_id = "redstone_relay_" .. tostring(i + 32)
    local block = peripheral.wrap(block_id)
    if block == nil then
        printError("Failed to find argument b output peripheral '" .. block_id .. "'")
        goto program_end
    end
    argument_b_outputs[i] = { block = block, side = "top" }
end

-- Get program path from args.
local args = {...}
if table.getn(args) < 1 then
    printError("ERROR: No program specified")
    goto program_end
end
local program_path = shell.resolve(args[1])
print("Loading program '" .. program_path .. "'")

-- Check if program exists
if not fs.exists(program_path) then
    printError("ERROR: Specified program doesn't exist")
    goto program_end
end

-- Load the program
local program_file = fs.open(program_path, "rb")
local instructions = {}
local arguments_a = {}
local arguments_b = {}
for i_instruction = 0, 65535 do
    local part_0 = program_file.read() or 0
    local part_1 = program_file.read() or 0
    local part_2 = program_file.read() or 0
    local part_3 = program_file.read() or 0
    local part_4 = program_file.read() or 0
    local part_5 = program_file.read() or 0
    local part_6 = program_file.read() or 0
    local part_7 = program_file.read() or 0
    -- if part_0 == nil or part_1 == nil or part_2 == nil or part_3 == nil or part_4 == nil or part_5 == nil or part_6 == nil or part_7 == nil then
    --     printError("Failed to load instruction " .. i_instruction)
    --     goto program_end
    -- end
    instructions[i_instruction] = bit.bor(bit.blshift(part_1, 8), part_0)
    arguments_a[i_instruction] = bit.bor(bit.blshift(part_3, 8), part_2)
    arguments_b[i_instruction] = bit.bor(bit.blshift(part_5, 8), part_4)
end
program_file.close()

-- Initialize inputs
for i = 0, 15, 1 do
    local input = pc_inputs[i]
    if input.block ~= nil then
        input.block.setOutput(input.side, false)
    end
end

-- Initialize outputs
for i = 0, 15, 1 do
    local output = instruction_outputs[i]
    output.block.setOutput(output.side, false)
end
for i = 0, 15, 1 do
    local output = argument_a_outputs[i]
    output.block.setOutput(output.side, false)
end
for i = 0, 15, 1 do
    local output = argument_b_outputs[i]
    output.block.setOutput(output.side, false)
end

-- Main Loop
print("Started program")
while true do
    -- Input program counter value.
    local pc = 0
    for i_input = 0, 15 do
        local input = pc_inputs[i_input]

        local value = input.block.getInput(input.side)
        if value == nil then
            printError("Failed to read bit " .. i_input)
            goto program_end
        end
        if value then
            pc = bit.bor(pc, bit.blshift(1, i_input))
        end
    end

    -- Output instruction value
    local instruction = instructions[pc]
    for i_output = 0, 15 do
        local value = bit.band(bit.blogic_rshift(instruction, i_output), 1) ~= 0
        local output = instruction_outputs[i_output]
        output.block.setOutput(output.side, value)
    end
    local argument_a = arguments_a[pc]
    for i_output = 0, 15 do
        local value = bit.band(bit.blogic_rshift(argument_a, i_output), 1) ~= 0
        local output = argument_a_outputs[i_output]
        output.block.setOutput(output.side, value)
    end
    local argument_b = arguments_b[pc]
    for i_output = 0, 15 do
        local value = bit.band(bit.blogic_rshift(argument_b, i_output), 1) ~= 0
        local output = argument_b_outputs[i_output]
        output.block.setOutput(output.side, value)
    end

    -- Wait 1 tick
    sleep(0.05)
end

::program_end::