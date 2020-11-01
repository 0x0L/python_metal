import numpy as np

import objc
import Metal

device = Metal.MTLCreateSystemDefaultDevice()

src = open('add.metal').read()
opts = Metal.MTLCompileOptions.new()
library = device.newLibraryWithSource_options_error_(src, opts, objc.NULL)[0]

# xcrun -sdk macosx metal -c add.metal
# xcrun -sdk macosx metallib add.air
# library = device.newLibraryWithFile_error_('default.metallib', None)[0]

addFunction = library.newFunctionWithName_('add_arrays')
addFunctionPSO = device.newComputePipelineStateWithFunction_error_(addFunction, objc.NULL)[0]

commandQueue = device.newCommandQueue()

arrayLength = 1 << 24
bufferSize = arrayLength * 4

bufferA = device.newBufferWithLength_options_(bufferSize, Metal.MTLResourceStorageModeShared)
bufferB = device.newBufferWithLength_options_(bufferSize, Metal.MTLResourceStorageModeShared)
bufferResult = device.newBufferWithLength_options_(bufferSize, Metal.MTLResourceStorageModeShared)

# TODO: fill A and B
vA = np.random.randn(arrayLength).astype(np.float32)
t = bufferA.contents()
m = t.as_buffer(bufferSize)
m[:] = bytes(vA)

vB = np.random.randn(arrayLength).astype(np.float32)
t = bufferB.contents()
m = t.as_buffer(bufferSize)
m[:] = bytes(vB)

commandBuffer = commandQueue.commandBuffer()
computeEncoder = commandBuffer.computeCommandEncoder()

computeEncoder.setComputePipelineState_(addFunctionPSO)
computeEncoder.setBuffer_offset_atIndex_(bufferA, 0, 0)
computeEncoder.setBuffer_offset_atIndex_(bufferB, 0, 1)
computeEncoder.setBuffer_offset_atIndex_(bufferResult, 0, 2)

gridSize = Metal.MTLSizeMake(arrayLength, 1, 1)

threadGroupSize = addFunctionPSO.maxTotalThreadsPerThreadgroup()
if threadGroupSize > arrayLength:
    threadGroupSize = arrayLength

threadgroupSize = Metal.MTLSizeMake(threadGroupSize, 1, 1)
computeEncoder.dispatchThreads_threadsPerThreadgroup_(gridSize, threadgroupSize)
computeEncoder.endEncoding()

commandBuffer.commit()
commandBuffer.waitUntilCompleted()

t = bufferResult.contents()
result = np.frombuffer(t.as_buffer(bufferSize), dtype=np.float32)

assert np.allclose(result, vA + vB)
