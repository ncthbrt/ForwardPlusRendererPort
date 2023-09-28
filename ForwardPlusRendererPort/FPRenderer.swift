//
//  Renderer.swift
//  ForwardPlusRendererPort
//
//  Created by Natalie Cuthbert on 2023/09/20.
//

// Our platform independent renderer class

import Metal
import MetalKit
import simd
// The 256 byte aligned size of our uniform structure


func bridge<T : AnyObject>(obj : T) -> UnsafeRawPointer {
    return UnsafeRawPointer(Unmanaged.passUnretained(obj).toOpaque())
}

func bridge<T : AnyObject>(ptr : UnsafeRawPointer) -> T {
    return Unmanaged<T>.fromOpaque(ptr).takeUnretainedValue()
}

func bridgeRetained<T : AnyObject>(obj : T) -> UnsafeRawPointer {
    return UnsafeRawPointer(Unmanaged.passRetained(obj).toOpaque())
}

func bridgeTransfer<T : AnyObject>(ptr : UnsafeRawPointer) -> T {
    return Unmanaged<T>.fromOpaque(ptr).takeRetainedValue()
}



let maxBuffersInFlight = 3

let FPNumFairyVertices = 7

enum RendererError: Error {
    case badVertexDescriptor
}

let FPDepthDataPixelFormat = MTLPixelFormat.r32Float;

let FPDepthBufferPixelFormat = MTLPixelFormat.depth32Float;


class FPRenderer: NSObject, MTKViewDelegate {
    
    var drawableSize: CGSize = CGSizeZero

    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    let vertexDescriptor: MTLVertexDescriptor
     
    let depthPrePassPipelineState: MTLRenderPipelineState
    let lightBinCreationPipelineState: MTLRenderPipelineState
    let lightCullingPipelineState: MTLRenderPipelineState
    let forwardLightingPipelineState: MTLRenderPipelineState
    let fairyPipelineState: MTLRenderPipelineState
    
    let depthState: MTLDepthStencilState
    let relaxedDepthState: MTLDepthStencilState
    let dontWriteDepthState: MTLDepthStencilState

    var frameDataBuffers: [MTLBuffer] = Array()
    
    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    
    var lightWorldPositions: [MTLBuffer] = Array()
    var lightEyePositions: [MTLBuffer] = Array()
    
    var nearPlane: Float = 1.0
    var farPlane: Float = 1500.0
    var fov: Float = 65.0 * (.pi / 180.0)
    
    var projectionMatrix: matrix_float4x4 = matrix_float4x4()
    
    var rotation: Float = 0
    
    let meshes: [FPMesh]
    let lightsData: MTLBuffer
    let fairy: MTLBuffer
    let viewRenderPassDescriptor: MTLRenderPassDescriptor

    init?(metalKitView: MTKView) {
        self.device = metalKitView.device!
        // This sample requires APIs avaliable with the MTLGPUFamilyApple4 feature set
        //   (which is avaliable on iOS devices with A11 GPUs or later or Macs with Apple Silicon).
        //   If the MTLGPUFamilyApple4 feature set is unavaliable, you would need to implement a
        //   backup path that does not use many of the APIs demonstrated in this sample.
        assert(self.device.supportsFamily(.apple4),
                 "Sample requires MTLGPUFamilyApple4 (on available on Macs with Apple Silicon or iOS devices with an A11 or later)")
        
        guard let library = device.makeDefaultLibrary() else { return nil }
        
        let storageMode = MTLResourceOptions.storageModeShared
        self.frameDataBuffers.reserveCapacity(maxBuffersInFlight)
        for i in 0..<maxBuffersInFlight {
            let idxStr = String(i)
            
            self.frameDataBuffers.append(device.makeBuffer(length: MemoryLayout<FPFrameData>.size, options: storageMode)!)
            self.frameDataBuffers[i].label = "FrameDataBuffer"+idxStr
            
            self.lightWorldPositions.append(device.makeBuffer(length: MemoryLayout<vector_float4>.size*Int(FPNumLights), options:storageMode)!)

            self.lightWorldPositions[i].label = "LightPositions"+idxStr

            self.lightEyePositions.append(device.makeBuffer(length: MemoryLayout<vector_float4>.size*Int(FPNumLights), options:storageMode)!)

            self.lightEyePositions[i].label = "LightPositions"+idxStr
        }
        
        // Create a vertex descriptor for the Metal pipeline. Specify the layout of vertices the
        //   pipeline should expect. The layout below keeps attributes used to calculate vertex shader
        //   output position (world position, skinning, tweening weights) separate from other
        //   attributes (texture coordinates, normals). This generally maximizes pipeline efficiency.

        self.vertexDescriptor = MTLVertexDescriptor()

        // Positions.
        vertexDescriptor.attributes[FPVertexAttributes.position.rawValue].format = .float3
        vertexDescriptor.attributes[FPVertexAttributes.position.rawValue].offset = 0
        vertexDescriptor.attributes[FPVertexAttributes.position.rawValue].bufferIndex = FPBufferIndices.indexMeshPositions.rawValue

        // Texture coordinates.
        vertexDescriptor.attributes[FPVertexAttributes.texcoord.rawValue].format = .float2
        vertexDescriptor.attributes[FPVertexAttributes.texcoord.rawValue].offset = 0
        vertexDescriptor.attributes[FPVertexAttributes.texcoord.rawValue].bufferIndex = FPBufferIndices.indexMeshGenerics.rawValue

        // Normals.
        vertexDescriptor.attributes[FPVertexAttributes.normal.rawValue].format = .half4
        vertexDescriptor.attributes[FPVertexAttributes.normal.rawValue].offset = 8
        vertexDescriptor.attributes[FPVertexAttributes.normal.rawValue].bufferIndex = FPBufferIndices.indexMeshGenerics.rawValue

        // Tangents.
        vertexDescriptor.attributes[FPVertexAttributes.tangent.rawValue].format = .half4
        vertexDescriptor.attributes[FPVertexAttributes.tangent.rawValue].offset = 16
        vertexDescriptor.attributes[FPVertexAttributes.tangent.rawValue].bufferIndex = FPBufferIndices.indexMeshGenerics.rawValue

        // Bitangents.
        vertexDescriptor.attributes[FPVertexAttributes.bitangent.rawValue].format = .half4
        vertexDescriptor.attributes[FPVertexAttributes.bitangent.rawValue].offset = 24
        vertexDescriptor.attributes[FPVertexAttributes.bitangent.rawValue].bufferIndex = FPBufferIndices.indexMeshGenerics.rawValue

        // Position buffer layout.
        vertexDescriptor.layouts[FPBufferIndices.indexMeshPositions.rawValue].stride = 12

        // Generic attribute buffer layout.
        vertexDescriptor.layouts[FPBufferIndices.indexMeshGenerics.rawValue].stride = 32

        metalKitView.colorPixelFormat = .bgra8Unorm_srgb

        // Set view's depth stencil pixel format to Invalid.  This app will manually manage it's own
        // depth buffer, not depend on the depth buffer managed by MTKView
        metalKitView.depthStencilPixelFormat = .invalid;

        // Create a render pipeline state descriptor.
        let renderPipelineStateDescriptor = MTLRenderPipelineDescriptor()

        renderPipelineStateDescriptor.rasterSampleCount = Int(FPNumSamples)
        renderPipelineStateDescriptor.vertexDescriptor = vertexDescriptor

        renderPipelineStateDescriptor.colorAttachments[FPRenderTargetIndices.lighting.rawValue].pixelFormat = metalKitView.colorPixelFormat
        renderPipelineStateDescriptor.colorAttachments[FPRenderTargetIndices.depth.rawValue].pixelFormat = FPDepthDataPixelFormat

        renderPipelineStateDescriptor.depthAttachmentPixelFormat = FPDepthBufferPixelFormat
        
        
        // Set unique descriptor values for the depth pre-pass pipeline state.
        let depthPrePassVertexFunction = library.makeFunction(name: "depth_pre_pass_vertex")
        let depthPrePassFragmentFunction = library.makeFunction(name: "depth_pre_pass_fragment")
        renderPipelineStateDescriptor.label = "Depth Pre-Pass"
        renderPipelineStateDescriptor.vertexDescriptor = vertexDescriptor
        renderPipelineStateDescriptor.vertexFunction = depthPrePassVertexFunction
        renderPipelineStateDescriptor.fragmentFunction = depthPrePassFragmentFunction

        try! self.depthPrePassPipelineState = device.makeRenderPipelineState(descriptor: renderPipelineStateDescriptor)

        // Set unique descriptor values for the standard material pipeline state.
        
        let vertexStandardMaterial = library.makeFunction(name: "forward_lighting_vertex")
        let fragmentStandardMaterial = library.makeFunction(name: "forward_lighting_fragment")

        renderPipelineStateDescriptor.label = "Forward Lighting"
        renderPipelineStateDescriptor.vertexDescriptor = vertexDescriptor
        renderPipelineStateDescriptor.vertexFunction = vertexStandardMaterial
        renderPipelineStateDescriptor.fragmentFunction = fragmentStandardMaterial
        try! self.forwardLightingPipelineState = device.makeRenderPipelineState(descriptor: renderPipelineStateDescriptor)
    

        // Set unique descriptor values for the fairy pipeline state.
        
        let fairyVertexFunction = library.makeFunction(name: "fairy_vertex")
        let fairyFragmentFunction = library.makeFunction(name:"fairy_fragment")

        renderPipelineStateDescriptor.label = "Fairy"
        renderPipelineStateDescriptor.vertexDescriptor = nil
        renderPipelineStateDescriptor.vertexFunction = fairyVertexFunction
        renderPipelineStateDescriptor.fragmentFunction = fairyFragmentFunction
        try! self.fairyPipelineState = device.makeRenderPipelineState(descriptor: renderPipelineStateDescriptor)
        
        let binCreationKernel = library.makeFunction(name:"create_bins")!

        let binCreationPipelineDescriptor = MTLTileRenderPipelineDescriptor()
        binCreationPipelineDescriptor.label = "Light Bin Creation"
        binCreationPipelineDescriptor.rasterSampleCount = Int(FPNumSamples)
        binCreationPipelineDescriptor.colorAttachments[FPRenderTargetIndices.lighting.rawValue].pixelFormat = metalKitView.colorPixelFormat
        binCreationPipelineDescriptor.colorAttachments[FPRenderTargetIndices.depth.rawValue].pixelFormat = FPDepthDataPixelFormat
        binCreationPipelineDescriptor.threadgroupSizeMatchesTileSize = true
        binCreationPipelineDescriptor.tileFunction = binCreationKernel
        try! self.lightBinCreationPipelineState = device.makeRenderPipelineState(tileDescriptor: binCreationPipelineDescriptor, options: MTLPipelineOption(), reflection: nil)


        // Create a tile render pipeline state descriptor for the culling pipeline state.
       
        let tileRenderPipelineDescriptor = MTLTileRenderPipelineDescriptor()
        let lightCullingKernel = library.makeFunction(name: "cull_lights")
        tileRenderPipelineDescriptor.tileFunction = lightCullingKernel!
        tileRenderPipelineDescriptor.label = "Light Culling"
        tileRenderPipelineDescriptor.rasterSampleCount = Int(FPNumSamples)

        tileRenderPipelineDescriptor.colorAttachments[FPRenderTargetIndices.lighting.rawValue].pixelFormat = metalKitView.colorPixelFormat
        tileRenderPipelineDescriptor.colorAttachments[FPRenderTargetIndices.depth.rawValue].pixelFormat = FPDepthDataPixelFormat

        tileRenderPipelineDescriptor.threadgroupSizeMatchesTileSize = true;
        try! self.lightCullingPipelineState = device.makeRenderPipelineState(tileDescriptor: tileRenderPipelineDescriptor, options:MTLPipelineOption(), reflection:nil)
    

        let depthStateDesc = MTLDepthStencilDescriptor()

        // Create a depth state with depth buffer write enabled.
        
        // Use `MTLCompareFunctionLess` because you render on a clean depth buffer.
        depthStateDesc.depthCompareFunction = .less
        depthStateDesc.isDepthWriteEnabled = true
        self.depthState = device.makeDepthStencilState(descriptor: depthStateDesc)!
    
        // Create a depth state with depth buffer write disabled and set the comparison function to
        //   `MTLCompareFunctionLessEqual`.
        
        // The comparison function is `MTLCompareFunctionLessEqual` instead of `MTLCompareFunctionLess`.
        //   The geometry pass renders to a pre-populated depth buffer (depth pre-pass) so each
        //   fragment needs to pass if its z-value is equal to the existing value already in the
        //   depth buffer.
        depthStateDesc.depthCompareFunction = .lessEqual
        depthStateDesc.isDepthWriteEnabled = false
        self.relaxedDepthState = device.makeDepthStencilState(descriptor: depthStateDesc)!
    

        // Create a depth state with depth buffer write disabled.
        depthStateDesc.depthCompareFunction = .less
        depthStateDesc.isDepthWriteEnabled = false
        self.dontWriteDepthState = device.makeDepthStencilState(descriptor: depthStateDesc)!
    

        // Create a render pass descriptor to render to the drawable
        
        self.viewRenderPassDescriptor = MTLRenderPassDescriptor();
        self.viewRenderPassDescriptor.colorAttachments[FPRenderTargetIndices.lighting.rawValue].loadAction = .clear
        self.viewRenderPassDescriptor.colorAttachments[FPRenderTargetIndices.depth.rawValue].loadAction = .clear
        self.viewRenderPassDescriptor.colorAttachments[FPRenderTargetIndices.depth.rawValue].storeAction = .dontCare
        self.viewRenderPassDescriptor.depthAttachment.loadAction = .clear
        self.viewRenderPassDescriptor.depthAttachment.storeAction = .dontCare
        self.viewRenderPassDescriptor.stencilAttachment.loadAction = .clear
        self.viewRenderPassDescriptor.stencilAttachment.storeAction = .dontCare
        self.viewRenderPassDescriptor.depthAttachment.clearDepth = 1.0
        self.viewRenderPassDescriptor.stencilAttachment.clearStencil = 0

        self.viewRenderPassDescriptor.tileWidth = Int(FPTileWidth)
        self.viewRenderPassDescriptor.tileHeight = Int(FPTileHeight)
        
        let bufferSize = max(Int(FPMaxLightsPerTile)*Int(MemoryLayout<Int32>.size), Int(FPTileWidth) * Int(FPTileHeight) * Int(MemoryLayout<Int32>.size))
        self.viewRenderPassDescriptor.threadgroupMemoryLength = Int(bufferSize) + Int(FPTileDataSize)

        if(FPNumSamples > 1)
        {
            self.viewRenderPassDescriptor.colorAttachments[FPRenderTargetIndices.lighting.rawValue].storeAction = .multisampleResolve
        }
        else
        {
            self.viewRenderPassDescriptor.colorAttachments[FPRenderTargetIndices.lighting.rawValue].storeAction = .store
        }
        
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue

        
        // Starting to load assets
        
        // Creata a Model I/O vertex descriptor so that the format and layout of Model I/O mesh vertices
        //   fits the Metal render pipeline's vertex descriptor layout.
        let modelIOVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(vertexDescriptor)
        
        // Indicate how each Metal vertex descriptor attribute maps to each Model I/O attribute.
        modelIOVertexDescriptor.attributes[FPVertexAttributes.position.rawValue]  = MDLVertexAttributePosition
        modelIOVertexDescriptor.attributes[FPVertexAttributes.texcoord.rawValue]  = MDLVertexAttributeTextureCoordinate
        modelIOVertexDescriptor.attributes[FPVertexAttributes.normal.rawValue]    = MDLVertexAttributeNormal
        modelIOVertexDescriptor.attributes[FPVertexAttributes.tangent.rawValue]   = MDLVertexAttributeTangent
        modelIOVertexDescriptor.attributes[FPVertexAttributes.bitangent.rawValue] = MDLVertexAttributeBitangent

        let modelFileURL = Bundle.main.url(forResource: "Meshes/Temple.obj", withExtension: nil)
        assert(modelFileURL != nil, "Could not find model Temple.obj file in bundle")

        // Create a MetalKit mesh buffer allocator so that ModelIO  will load mesh data directly into
        //   Metal buffers accessible by the GPU

        do {
            self.meshes = try FPMesh.load(url: modelFileURL!, modelIOVertexDescriptor: modelIOVertexDescriptor, device: device)
        } catch {
            self.meshes = Array()
        }
        
        guard let lightData = device.makeBuffer(length: MemoryLayout<FPPointLight>.size*Int(FPNumLights), options: .storageModeShared) else {
            return nil
        }
        self.lightsData = lightData
        self.lightsData.label = "LightData"
        
        
        // Create a simple 2D triangle strip circle mesh for the fairies.
        let fairySize: Float = 2.5
        var fairyVertices: [FPSimpleVertex] =  Array(repeating: FPSimpleVertex(), count: FPNumFairyVertices)
        let angle = 2.0 * .pi / Float(FPNumFairyVertices);
        for vtx in 0..<FPNumFairyVertices
        {
            let point = (vtx % 2 == 0) ? (vtx + 1) / 2 : -vtx / 2
            fairyVertices[vtx].position =  vector_float2(x: sin(Float(point)*angle), y: cos(Float(point)*angle))
            fairyVertices[vtx].position *= fairySize;
        }

        guard let fairy = device.makeBuffer(bytes: fairyVertices, length: MemoryLayout.size(ofValue: fairyVertices), options: MTLResourceOptions()) else {
            return nil
        }
        
        self.fairy = fairy
    
        
        // Populating lights
        let lightDataContents = lightsData.contents()
        let lightWorldPositionContents = lightWorldPositions[0].contents()

        srandom(0x134e5348)

        for lightId in 0..<FPNumLights
        {
            var distance:Float = 0
            var height:Float = 0
            var angle:Float = 0
            if(lightId < FPNumLights/4) {
                distance = Float.random(in: 140...260);
                height = Float.random(in: 140...150);
                angle = Float.random(in: 0...(.pi*2));
            } else if(lightId < (FPNumLights*3)/4) {
                distance = Float.random(in: 350...362)
                height = Float.random(in:140...400);
                angle = Float.random(in: 0...(.pi*2))
            } else if(lightId < (FPNumLights*15)/16) {
                distance = Float.random(in:400...480)
                height = Float.random(in:68...80);
                angle = Float.random(in: 0...(.pi*2))
            } else {
                distance = 40;
                height = Float.random(in:220...350);
                angle = Float.random(in: 0...(.pi*2))
            }
            
            var lightData = FPPointLight()
            
            let colorId = Int.random(in: 0..<3)
            if(colorId == 0) {
                lightData.lightColor = vector_float3(Float.random(in: 2...3),Float.random(in:0...2),Float.random(in: 0...2))
            } else if ( colorId == 1) {
                lightData.lightColor = vector_float3(Float.random(in: 0...2),Float.random(in:2...3),Float.random(in: 0...2))
            } else {
                lightData.lightColor = vector_float3(Float.random(in: 0...2),Float.random(in:0...2),Float.random(in: 2...3))
            }

            lightData.lightRadius = Float.random(in:25...35)
            lightData.lightSpeed = Float.random(in:0.003...0.015)
            
            let lightPosition = vector_float4(distance*sinf(angle),height,distance*cosf(angle),lightData.lightRadius)
            
            lightDataContents.storeBytes(of: lightData, toByteOffset: MemoryLayout<FPPointLight>.size*Int(lightId), as: FPPointLight.self)
            lightWorldPositionContents.storeBytes(of: lightPosition, toByteOffset: MemoryLayout<vector_float4>.size*Int(lightId), as: vector_float4.self)
        }

        memcpy(lightWorldPositions[1].contents(), lightWorldPositions[0].contents(), Int(FPNumLights) * MemoryLayout<vector_float3>.size)
        memcpy(lightWorldPositions[2].contents(), lightWorldPositions[0].contents(), Int(FPNumLights) * MemoryLayout<vector_float3>.size)

        super.init()
    }
    
    
//    class func buildRenderPipelineWithDevice(device: MTLDevice,
//                                             metalKitView: MTKView,
//                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
//        /// Build a render state pipeline object
//        
//        let library = device.makeDefaultLibrary()
//        
//        let vertexFunction = library?.makeFunction(name: "vertexShader")
//        let fragmentFunction = library?.makeFunction(name: "fragmentShader")
//        
//        let pipelineDescriptor = MTLRenderPipelineDescriptor()
//        pipelineDescriptor.label = "RenderPipeline"
//        pipelineDescriptor.rasterSampleCount = metalKitView.sampleCount
//        pipelineDescriptor.vertexFunction = vertexFunction
//        pipelineDescriptor.fragmentFunction = fragmentFunction
//        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
//        
//        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
//        pipelineDescriptor.depthAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
//        pipelineDescriptor.stencilAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
//        
//        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
//    }
    
//    class func buildMesh(device: MTLDevice,
//                         mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
//        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor
//        
//        let metalAllocator = MTKMeshBufferAllocator(device: device)
//        
//        let mdlMesh = MDLMesh.newBox(withDimensions: SIMD3<Float>(4, 4, 4),
//                                     segments: SIMD3<UInt32>(2, 2, 2),
//                                     geometryType: MDLGeometryType.triangles,
//                                     inwardNormals:false,
//                                     allocator: metalAllocator)
//        
//        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)
//        
//        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
//            throw RendererError.badVertexDescriptor
//        }
//        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
//        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
//        
//        mdlMesh.vertexDescriptor = mdlVertexDescriptor
//        
//        return try MTKMesh(mesh:mdlMesh, device:device)
//    }
//    
//    class func loadTexture(device: MTLDevice,
//                           textureName: String) throws -> MTLTexture {
//        /// Load texture data with optimal parameters for sampling
//        
//        let textureLoader = MTKTextureLoader(device: device)
//        
//        let textureLoaderOptions = [
//            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
//            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
//        ]
//        
//        return try textureLoader.newTexture(name: textureName,
//                                            scaleFactor: 1.0,
//                                            bundle: nil,
//                                            options: textureLoaderOptions)
//        
//    }
    
    private func updateDynamicBufferState() {
//        /// Update the state of our uniform buffers before rendering
//        
//        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight
//        
//        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex
//        
//        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:Uniforms.self, capacity:1)
    }
//    
    private func updateGameState() {
        /// Update any game state before rendering
//        
//        uniforms[0].projectionMatrix = projectionMatrix
//        
//        let rotationAxis = SIMD3<Float>(1, 1, 0)
//        let modelMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
//        let viewMatrix = matrix4x4_translation(0.0, 0.0, -8.0)
//        uniforms[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
//        rotation += 0.01
    }
    
    func draw(in view: MTKView) {
        /// Per frame updates hare
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            
            let semaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
                semaphore.signal()
            }
            
            self.updateDynamicBufferState()
            
            self.updateGameState()
            
            /// Delay getting the currentRenderPassDescriptor until we absolutely need it to avoid
            ///   holding onto the drawable and blocking the display pipeline any longer than necessary
            let renderPassDescriptor = view.currentRenderPassDescriptor
            
            if let renderPassDescriptor = renderPassDescriptor, let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                
                /// Final pass rendering code here
//                renderEncoder.label = "Primary Render Encoder"
//                
//                renderEncoder.pushDebugGroup("Draw Box")
//                
//                renderEncoder.setCullMode(.back)
//                
//                renderEncoder.setFrontFacing(.counterClockwise)
//                
//                renderEncoder.setRenderPipelineState(pipelineState)
//                
//                renderEncoder.setDepthStencilState(depthState)
//                
//                renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
//                renderEncoder.setFragmentBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
//                
//                for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
//                    guard let layout = element as? MDLVertexBufferLayout else {
//                        return
//                    }
//                    
//                    if layout.stride != 0 {
//                        let buffer = mesh.vertexBuffers[index]
//                        renderEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
//                    }
//                }
//                
//                renderEncoder.setFragmentTexture(colorMap, index: TextureIndex.color.rawValue)
//                
//                for submesh in mesh.submeshes {
//                    renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
//                                                        indexCount: submesh.indexCount,
//                                                        indexType: submesh.indexType,
//                                                        indexBuffer: submesh.indexBuffer.buffer,
//                                                        indexBufferOffset: submesh.indexBuffer.offset)
//                    
//                }
//                
//                renderEncoder.popDebugGroup()
                
                renderEncoder.endEncoding()
                
                if let drawable = view.currentDrawable {
                    commandBuffer.present(drawable)
                }
            }
            
            commandBuffer.commit()
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        /// Respond to drawable size or orientation changes here
        
//        let aspect = Float(size.width) / Float(size.height)
//        projectionMatrix = matrix_perspective_right_hand(fovyRadians: radians_from_degrees(65), aspectRatio:aspect, nearZ: 0.1, farZ: 100.0)
        
        self.drawableSize = size

        // Update the aspect ratio and projection matrix because the view orientation
        //   or size has changed.
        let aspect = Float(size.width) / Float(size.height)
        self.fov = 65 * (.pi / 180.0)
        self.nearPlane = 1.0
        self.farPlane = 1500.0
        self.projectionMatrix = matrix_perspective_left_hand(fovyRadians: self.fov, aspectRatio: aspect, nearZ: nearPlane, farZ: farPlane);

        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.width = Int(self.drawableSize.width)
        textureDescriptor.height = Int(self.drawableSize.height)
        textureDescriptor.usage = .renderTarget
        textureDescriptor.storageMode = .memoryless
//
//        if(FPNumSamples > 1)
//        {
//            self.textureDescriptor.sampleCount = AAPLNumSamples;
//            textureDescriptor.textureType = MTLTextureType2DMultisample;
//            textureDescriptor.pixelFormat = view.colorPixelFormat;
//
//            id<MTLTexture> msaaColorTexture = [_device newTextureWithDescriptor:textureDescriptor];
//
//            _viewRenderPassDescriptor.colorAttachments[AAPLRenderTargetLighting].texture = msaaColorTexture;
//        }
//        else
//        {
        textureDescriptor.textureType = .type2D;
//        }
        
        // Create depth buffer texture for depth testing
        textureDescriptor.pixelFormat = FPDepthBufferPixelFormat;
        let depthBufferTexture = self.device.makeTexture(descriptor: textureDescriptor)
        self.viewRenderPassDescriptor.depthAttachment.texture = depthBufferTexture;
    
        // Create depth data texture to determine min max depth for each tile

        textureDescriptor.pixelFormat = FPDepthDataPixelFormat;
        let depthDataTexture = self.device.makeTexture(descriptor: textureDescriptor);
        self.viewRenderPassDescriptor.colorAttachments[1].texture = depthDataTexture;
    }
}

// Generic matrix math utility functions
func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return matrix_float4x4(columns:(vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                         vector_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                         vector_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                         vector_float4(                  0,                   0,                   0, 1)))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return matrix_float4x4(columns:(vector_float4(1, 0, 0, 0),
                                         vector_float4(0, 1, 0, 0),
                                         vector_float4(0, 0, 1, 0),
                                         vector_float4(translationX, translationY, translationZ, 1)))
}

func matrix_perspective_right_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4(columns:(vector_float4(xs,  0, 0,   0),
                                         vector_float4( 0, ys, 0,   0),
                                         vector_float4( 0,  0, zs, -nearZ * zs),
                                         vector_float4( 0,  0, 1, 0)))
}


func matrix_perspective_left_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4(columns:(vector_float4(xs,  0, 0,   0),
                                         vector_float4( 0, ys, 0,   0),
                                         vector_float4( 0,  0, zs, -1),
                                         vector_float4( 0,  0, zs * nearZ, 0)))
}




func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}
