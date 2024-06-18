#include "Volumetrics.h"
#include "deferred.h"
#include "VulkanTools.h"
#include "VulkanInitializers.hpp"


void VulkanVolumetrics::Init(VulkanExample* example, vks::VulkanDevice* device, Camera* camera, VkQueue* pQueue)
{
	// Store a member pointer to the device
	pDevice = device;
	pExampleBase = example;
	pCamera = camera;

	XWindSpeed = 0.4f;
	YWindSpeed = 0.7f;
	ZWindSpeed = -0.1f;

	FogShapesData.Spheres[0].Pos = glm::vec3(1.25f, 5.4f, -2.01f);
	FogShapesData.Spheres[0].Radius = 2.9f;
	FogShapesData.Spheres[1].Pos = glm::vec3(-0.8f, 2.5f, -1.25f);
	FogShapesData.Spheres[1].Radius = 3.5f;
	FogShapesData.Spheres[2].Pos = glm::vec3(0.f, -2.5f, -2.f);
	FogShapesData.Spheres[2].Radius = 4.f;

	// Create the memory buffer for the sphere volume
	{
		device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &FogShapesBuff, sizeof(FogShapes), (void*)&FogShapesData);
	}

	VolumetricsData.Near = pCamera->getNearClip();
	VolumetricsData.Far = pCamera->getFarClip();
	VolumetricsData.Albedo = glm::vec3(0.8f, 0.8f, 0.8f);
	VolumetricsData.InitialStepSize = 0.025f;
	VolumetricsData.StepFallOff = 0.00f;
	VolumetricsData.LightMarchSize = 0.2f;
	VolumetricsData.Absorption = 0.5f;
	VolumetricsData.Density = 0.8f;
	VolumetricsData.AbsorptionCutoff = 0.01f;
	VolumetricsData.LightAbsorptionCutoff = 0.01f;
	VolumetricsData.NoiseXTile = 11.0f;
	VolumetricsData.NoiseYTile = 10.8f;
	VolumetricsData.NoiseZTile = 7.5f;
	VolumetricsData.NoiseXOffset = 0.0f;
	VolumetricsData.NoiseYOffset = 0.0f;
	VolumetricsData.NoiseZOffset = 0.0f;
	VolumetricsData.NoiseFactor = 1.5f;
	VolumetricsData.SmoothFactor = 0.9f;
	VolumetricsData.MapHeight = 360;
	VolumetricsData.MapWidth = 640;
	VolumetricsData.MapDepth = 16 * 16 * 2;

	// Create memory buffer for the volumetrics info
	{
		device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &VolumetricsBuff, sizeof(VolumetricsInfo), (void*)&VolumetricsData);
	}

	// Create memory buffer for the perlin noise, Setting the Sampler Address mode to mirrored repeat
	{
		PerlinNoise.loadFromFileCustomAddressMode(getAssetPath() + "Volumetrics/PerlinNoise512.ktx", VK_FORMAT_R8G8B8A8_UNORM, pDevice, *pQueue, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
	}

	PrepareTextures();

	PrepareDescriptors();

	PreparePipelines();

	BuildCommandBuffers();

	// Prepare common submit info parameters for later
	SubmitInfo = vks::initializers::submitInfo();
	SubmitInfo.pWaitDstStageMask = &SubmitStageFlag;
	SubmitInfo.commandBufferCount = 1;
	SubmitInfo.signalSemaphoreCount = 1;
}

void VulkanVolumetrics::PrepareTextures()
{
	// Get the compute queue for the device
	vkGetDeviceQueue(*pDevice, pDevice->queueFamilyIndices.compute, 0, &ComputeQueue);

	// Create 3D Texture Buffer for the first stage compute output
	{
		FirstStageTexture.device = pDevice;

		VkImageCreateInfo ImageCreateInfo = vks::initializers::imageCreateInfo();
		ImageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
		ImageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		// Set width and height to match screen size for now
		FirstStageTexture.width = VolumetricsData.MapWidth;
		ImageCreateInfo.extent.width = FirstStageTexture.width;
		FirstStageTexture.height = VolumetricsData.MapHeight;
		ImageCreateInfo.extent.height = FirstStageTexture.height;
		// Set depth to be equal to the number of max steps
		ImageCreateInfo.extent.depth = VolumetricsData.MapDepth;
		ImageCreateInfo.mipLevels = 1;
		ImageCreateInfo.arrayLayers = 1;
		ImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		ImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		ImageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		VK_CHECK_RESULT(vkCreateImage(*pDevice, &ImageCreateInfo, nullptr, &FirstStageTexture.image));
		vkGetImageMemoryRequirements(*pDevice, FirstStageTexture.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = pDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(*pDevice, &memAlloc, nullptr, &FirstStageTexture.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(*pDevice, FirstStageTexture.image, FirstStageTexture.deviceMemory, 0));

		VkCommandBuffer layoutCmd = pDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		FirstStageTexture.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		vks::tools::setImageLayout(layoutCmd, FirstStageTexture.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, FirstStageTexture.imageLayout);
		pDevice->flushCommandBuffer(layoutCmd, pExampleBase->queue, true);

		VkImageViewCreateInfo imageView = vks::initializers::imageViewCreateInfo();
		imageView.viewType = VK_IMAGE_VIEW_TYPE_3D;
		imageView.format = VK_FORMAT_R8G8B8A8_UNORM;
		imageView.subresourceRange = {};
		imageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageView.subresourceRange.baseMipLevel = 0;
		imageView.subresourceRange.levelCount = 1;
		imageView.subresourceRange.baseArrayLayer = 0;
		imageView.subresourceRange.layerCount = 1;
		imageView.image = FirstStageTexture.image;
		VK_CHECK_RESULT(vkCreateImageView(*pDevice, &imageView, nullptr, &FirstStageTexture.view));

		// Create a sampler for second stage to use
		VkSamplerCreateInfo samplerci = vks::initializers::samplerCreateInfo();
		samplerci.magFilter = VK_FILTER_NEAREST;
		samplerci.minFilter = VK_FILTER_NEAREST;
		samplerci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerci.addressModeV = samplerci.addressModeU;
		samplerci.addressModeW = samplerci.addressModeU;
		samplerci.mipLodBias = 0.0f;
		samplerci.maxAnisotropy = 1.0f;
		samplerci.minLod = 0.0f;
		samplerci.maxLod = 1.0f;
		samplerci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
		VK_CHECK_RESULT(vkCreateSampler(*pDevice, &samplerci, nullptr, &FirstStageTexture.sampler));

		FirstStageTexture.descriptor = vks::initializers::descriptorImageInfo(
			FirstStageTexture.sampler,
			FirstStageTexture.view,
			VK_IMAGE_LAYOUT_GENERAL); // TODO: Check if this is optimal
	}

	// Create 2D Texture Buffer for second stage compute output
	{
		SecondStageTexture.device = pDevice;

		VkImageCreateInfo ImageCreateInfo = vks::initializers::imageCreateInfo();
		ImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		ImageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		// Set width and height to match screen size for now
		SecondStageTexture.width = VolumetricsData.MapWidth;
		ImageCreateInfo.extent.width = SecondStageTexture.width;
		SecondStageTexture.height = VolumetricsData.MapHeight;
		ImageCreateInfo.extent.height = SecondStageTexture.height;
		ImageCreateInfo.extent.depth = 1;
		ImageCreateInfo.mipLevels = 1;
		ImageCreateInfo.arrayLayers = 1;
		ImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		ImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		ImageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		VK_CHECK_RESULT(vkCreateImage(*pDevice, &ImageCreateInfo, nullptr, &SecondStageTexture.image));
		vkGetImageMemoryRequirements(*pDevice, SecondStageTexture.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = pDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(*pDevice, &memAlloc, nullptr, &SecondStageTexture.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(*pDevice, SecondStageTexture.image, SecondStageTexture.deviceMemory, 0));

		VkCommandBuffer layoutCmd = pDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		SecondStageTexture.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		vks::tools::setImageLayout(layoutCmd, SecondStageTexture.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, SecondStageTexture.imageLayout);
		pDevice->flushCommandBuffer(layoutCmd, pExampleBase->queue, true);

		VkImageViewCreateInfo imageView = vks::initializers::imageViewCreateInfo();
		imageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageView.format = VK_FORMAT_R8G8B8A8_UNORM;
		imageView.subresourceRange = {};
		imageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageView.subresourceRange.baseMipLevel = 0;
		imageView.subresourceRange.levelCount = 1;
		imageView.subresourceRange.baseArrayLayer = 0;
		imageView.subresourceRange.layerCount = 1;
		imageView.image = SecondStageTexture.image;
		VK_CHECK_RESULT(vkCreateImageView(*pDevice, &imageView, nullptr, &SecondStageTexture.view));

		// Create a sampler for the lighting pass fragment shader to use
		VkSamplerCreateInfo samplerci = vks::initializers::samplerCreateInfo();
		samplerci.magFilter = VK_FILTER_NEAREST;
		samplerci.minFilter = VK_FILTER_NEAREST;
		samplerci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerci.addressModeV = samplerci.addressModeU;
		samplerci.addressModeW = samplerci.addressModeU;
		samplerci.mipLodBias = 0.0f;
		samplerci.maxAnisotropy = 1.0f;
		samplerci.minLod = 0.0f;
		samplerci.maxLod = 1.0f;
		samplerci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
		VK_CHECK_RESULT(vkCreateSampler(*pDevice, &samplerci, nullptr, &SecondStageTexture.sampler));

		SecondStageTexture.descriptor = vks::initializers::descriptorImageInfo(
			SecondStageTexture.sampler,
			SecondStageTexture.view,
			VK_IMAGE_LAYOUT_GENERAL); // TODO: Check if this is optimal
	}

}

void VulkanVolumetrics::PrepareDescriptors()
{
	// Create a descriptor pool for the compute stages
	{
		std::vector<VkDescriptorPoolSize> poolSizes = {
			//Scene info, Scene Fog & Volumetrics info
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4),
			// Noise texture sampler for the first stage, and 3D frustrum texture sampler for the second stage
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2),
			// Output 3D texture from the first compute stage, and the 2d texture output for the second
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4)
		};

		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, static_cast<uint32_t>(poolSizes.size())); // MAX SETS MAY NEED CHANGING
		VK_CHECK_RESULT(vkCreateDescriptorPool(pDevice->logicalDevice, &descriptorPoolInfo, nullptr, &DescPool));
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Descriptor Set layouts and Descriptor sets for the first compute stage
	{
		// Image descriptors for the offscreen color attachments
		VkDescriptorImageInfo PositionDesciptor =
			vks::initializers::descriptorImageInfo(
				pExampleBase->colorSampler,
				pExampleBase->offScreenFrameBuf.position.view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Scene Info
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1),
			// Fog Shape
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1),
			// Volumetrics Info
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1),
			// Perlin Noise sampler
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 3, 1),
			// Position sampler
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 4, 1),
			// 3D Output texture
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5, 1)
		};

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(pDevice->logicalDevice, &descriptorSetLayoutCI, nullptr, &ComputePipelines[0].DescSetLayout));

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&ComputePipelines[0].DescSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(pDevice->logicalDevice, &pipelineLayoutCreateInfo, nullptr, &ComputePipelines[0].PipelineLayout));

		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(DescPool, &ComputePipelines[0].DescSetLayout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(pDevice->logicalDevice, &allocInfo, &ComputePipelines[0].DescSet));


		std::vector<VkWriteDescriptorSet> writeDescriptorSets =
		{
			vks::initializers::writeDescriptorSet(ComputePipelines[0].DescSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &pExampleBase->uniformBuffers.composition.descriptor),
			vks::initializers::writeDescriptorSet(ComputePipelines[0].DescSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, &FogShapesBuff.descriptor),
			vks::initializers::writeDescriptorSet(ComputePipelines[0].DescSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2, &VolumetricsBuff.descriptor),
			vks::initializers::writeDescriptorSet(ComputePipelines[0].DescSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3, &PerlinNoise.descriptor),
			vks::initializers::writeDescriptorSet(ComputePipelines[0].DescSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4, &PositionDesciptor),
			vks::initializers::writeDescriptorSet(ComputePipelines[0].DescSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5, &FirstStageTexture.descriptor)
		};

		vkUpdateDescriptorSets(pDevice->logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
			writeDescriptorSets.data(), 0, nullptr);
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Descriptor Set layouts and Descriptor sets for the second compute stage
	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1),
			// 3D texture from previous compute stage
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1),
			// 2D Texture output
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1),
		};

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(pDevice->logicalDevice, &descriptorSetLayoutCI, nullptr, &ComputePipelines[1].DescSetLayout));

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&ComputePipelines[1].DescSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(pDevice->logicalDevice, &pipelineLayoutCreateInfo, nullptr, &ComputePipelines[1].PipelineLayout));

		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(DescPool, &ComputePipelines[1].DescSetLayout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(pDevice->logicalDevice, &allocInfo, &ComputePipelines[1].DescSet));


		std::vector<VkWriteDescriptorSet> writeDescriptorSets =
		{
			vks::initializers::writeDescriptorSet(ComputePipelines[1].DescSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &VolumetricsBuff.descriptor),
			vks::initializers::writeDescriptorSet(ComputePipelines[1].DescSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &FirstStageTexture.descriptor),
			vks::initializers::writeDescriptorSet(ComputePipelines[1].DescSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 ,&SecondStageTexture.descriptor)
		};

		vkUpdateDescriptorSets(pDevice->logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
			writeDescriptorSets.data(), 0, nullptr);
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Descriptor Set layouts and descriptor sets for binding the final volumetrics texture to the lighting pass
	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// 2D texture produced from the 2nd compute stage
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1)
		};

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(pDevice->logicalDevice, &descriptorSetLayoutCI, nullptr, &LightingPassDescSetLayout));


		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(DescPool, &LightingPassDescSetLayout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(pDevice->logicalDevice, &allocInfo, &LightingPassDescSet));


		std::vector<VkWriteDescriptorSet> writeDescriptorSets =
		{
			vks::initializers::writeDescriptorSet(LightingPassDescSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0, &SecondStageTexture.descriptor)
		};

		vkUpdateDescriptorSets(pDevice->logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
			writeDescriptorSets.data(), 0, nullptr);
	}
}

void VulkanVolumetrics::PreparePipelines()
{
	// Shared Command Pool for both compute pipelines

	VkCommandPoolCreateInfo cmdPoolInfo = {};
	cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmdPoolInfo.queueFamilyIndex = pDevice->queueFamilyIndices.compute;
	cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VK_CHECK_RESULT(vkCreateCommandPool(*pDevice, &cmdPoolInfo, nullptr, &ComputeCmdPool));

	// Create First Compute Stage Pipeline
	{
		VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(ComputePipelines[0].PipelineLayout, 0);

		// One pipeline for each available image filter
		computePipelineCreateInfo.stage = pExampleBase->loadShader(pExampleBase->getShadersPath() + "deferred/volumetrics_firststage.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

		VK_CHECK_RESULT(vkCreateComputePipelines(*pDevice, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &ComputePipelines[0].Pipeline));

		// Create a command buffer for compute operations
		VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(ComputeCmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VK_CHECK_RESULT(vkAllocateCommandBuffers(*pDevice, &cmdBufAllocateInfo, &ComputePipelines[0].CmdBuff));

		// Semaphore for compute & graphics sync
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(*pDevice, &semaphoreCreateInfo, nullptr, &ComputePipelines[0].Semaphore));
	}

	// Create First Compute Stage Pipeline
	{
		VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(ComputePipelines[1].PipelineLayout, 0);

		// One pipeline for each available image filter
		computePipelineCreateInfo.stage = pExampleBase->loadShader(pExampleBase->getShadersPath() + "deferred/volumetrics_secondstage.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

		VK_CHECK_RESULT(vkCreateComputePipelines(*pDevice, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &ComputePipelines[1].Pipeline));

		// Create a command buffer for compute operations
		VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(ComputeCmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VK_CHECK_RESULT(vkAllocateCommandBuffers(*pDevice, &cmdBufAllocateInfo, &ComputePipelines[1].CmdBuff));

		// Semaphore for compute & graphics sync
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(*pDevice, &semaphoreCreateInfo, nullptr, &ComputePipelines[1].Semaphore));
	}
	
}

void VulkanVolumetrics::BuildCommandBuffers()
{
	//Build the command buffer for the first stage compute pipeline
	{
		// Flush the queue if we're rebuilding the command buffer after a pipeline change to ensure it's not currently in use
		vkQueueWaitIdle(ComputeQueue);

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VK_CHECK_RESULT(vkBeginCommandBuffer(ComputePipelines[0].CmdBuff, &cmdBufInfo));

		vkCmdBindPipeline(ComputePipelines[0].CmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipelines[0].Pipeline);
		vkCmdBindDescriptorSets(ComputePipelines[0].CmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipelines[0].PipelineLayout, 0, 1, &ComputePipelines[0].DescSet, 0, 0);

		// Thread group sizes set to 8 x 8 x 8 in the compute shader, so we dispatch enough groups to cover the 3d map
		vkCmdDispatch(ComputePipelines[0].CmdBuff, FirstStageTexture.width / 8, FirstStageTexture.height / 8, VolumetricsData.MapDepth / 8);

		vkEndCommandBuffer(ComputePipelines[0].CmdBuff);
	}
	
	//Build the command buffer for the second stage compute pipeline
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VK_CHECK_RESULT(vkBeginCommandBuffer(ComputePipelines[1].CmdBuff, &cmdBufInfo));

		vkCmdBindPipeline(ComputePipelines[1].CmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipelines[1].Pipeline);
		vkCmdBindDescriptorSets(ComputePipelines[1].CmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipelines[1].PipelineLayout, 0, 1, &ComputePipelines[1].DescSet, 0, 0);

		// Thread group sizes set to 8 x 8 in the compute shader, so we dispatch enough groups to cover the 2D output texture
		vkCmdDispatch(ComputePipelines[1].CmdBuff, SecondStageTexture.width / 8, SecondStageTexture.height / 8, 1);

		vkEndCommandBuffer(ComputePipelines[1].CmdBuff);
	}


}

void VulkanVolumetrics::UpdateBuffers()
{
	float DTime = static_cast<float>(FrameTimer.total_elapsed()) / 1000.f;
	FrameTimer.restart();

	VolumetricsData.NoiseXOffset += DTime * XWindSpeed;
	VolumetricsData.NoiseXOffset = fmodf(VolumetricsData.NoiseXOffset, FLT_MAX - 10.f);
	VolumetricsData.NoiseYOffset += DTime * YWindSpeed;
	VolumetricsData.NoiseYOffset = fmodf(VolumetricsData.NoiseYOffset, FLT_MAX - 10.f);
	VolumetricsData.NoiseZOffset += DTime * ZWindSpeed;
	VolumetricsData.NoiseZOffset = fmodf(VolumetricsData.NoiseZOffset, FLT_MAX - 10.f);
	if (StepFallOffMult == 0.f)
	{
		VolumetricsData.StepFallOff = 0.f;
	}
	else
	{
		VolumetricsData.StepFallOff = StepFallOffMult / 100.f;
	}

	VolumetricsBuff.map();
	memcpy(VolumetricsBuff.mapped, &VolumetricsData, sizeof(VolumetricsData));
	VolumetricsBuff.unmap();
	FogShapesBuff.map();
	memcpy(FogShapesBuff.mapped, &FogShapesData, sizeof(FogShapes));
	FogShapesBuff.unmap();
	SettingsOOD = false;
}

VkSemaphore* VulkanVolumetrics::SubmitCommands(VkSemaphore* pWaitSemaphore)
{
	// Submit First Stage
	// If a semaphore was passed in wait on that
	SubmitInfo.pWaitSemaphores = pWaitSemaphore;
	SubmitInfo.waitSemaphoreCount = (pWaitSemaphore != VK_NULL_HANDLE)? 1 : 0;

	SubmitInfo.pSignalSemaphores = &ComputePipelines[0].Semaphore;

	SubmitInfo.pCommandBuffers = &ComputePipelines[0].CmdBuff;
	VK_CHECK_RESULT(vkQueueSubmit(ComputeQueue, 1, &SubmitInfo, VK_NULL_HANDLE));


	// Submit Second Stage

	// Wait on the previous stage to complete
	SubmitInfo.pWaitSemaphores = &ComputePipelines[0].Semaphore;
	SubmitInfo.waitSemaphoreCount = 1;

	SubmitInfo.pSignalSemaphores = &ComputePipelines[1].Semaphore;

	SubmitInfo.pCommandBuffers = &ComputePipelines[1].CmdBuff;
	VK_CHECK_RESULT(vkQueueSubmit(ComputeQueue, 1, &SubmitInfo, VK_NULL_HANDLE));

	return &ComputePipelines[1].Semaphore;
}

void VulkanVolumetrics::Release(VkDevice& device)
{
	// Release Second Stage Compute Pipeline resources
	vkDestroySemaphore(device, ComputePipelines[1].Semaphore, nullptr);
	
	vkDestroyPipeline(device, ComputePipelines[1].Pipeline, nullptr);
	vkDestroyPipelineLayout(device, ComputePipelines[1].PipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(device, ComputePipelines[1].DescSetLayout, nullptr);

	// Release First Stage Compute Pipeline resources
	vkDestroySemaphore(device, ComputePipelines[0].Semaphore, nullptr);
	vkDestroyPipeline(device, ComputePipelines[0].Pipeline, nullptr);
	vkDestroyPipelineLayout(device, ComputePipelines[0].PipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(device, ComputePipelines[0].DescSetLayout, nullptr);

	// Release Additional Descriptor set layout
	vkDestroyDescriptorSetLayout(device, LightingPassDescSetLayout, nullptr);

	// Release shared pools
	vkDestroyCommandPool(device, ComputeCmdPool, nullptr);
	vkDestroyDescriptorPool(device, DescPool, nullptr);

	// Release Textures and buffers
	SecondStageTexture.destroy();
	FirstStageTexture.destroy();

	PerlinNoise.destroy();

	VolumetricsBuff.destroy();

	FogShapesBuff.destroy();
}

static int s_CurrentIMGUISphere = 0;

void VulkanVolumetrics::UpdateOverlay(vks::UIOverlay* overlay)
{
	if (overlay->header("Fog Settings"))
	{

		overlay->text("Volumetrics Resolution:\n");
		overlay->text("Width: %d", VolumetricsData.MapWidth);
		overlay->text("Height: %d", VolumetricsData.MapHeight);
		overlay->text("Depth: %d", VolumetricsData.MapDepth);
		overlay->sliderFloat("Albedo R", &VolumetricsData.Albedo.r, 0.f, 1.f);
		overlay->sliderFloat("Albedo G", &VolumetricsData.Albedo.g, 0.f, 1.f);
		overlay->sliderFloat("Albedo B", &VolumetricsData.Albedo.b, 0.f, 1.f);

		overlay->sliderFloat("InitialStepSize", &VolumetricsData.InitialStepSize, 0.01f, 0.1f);
		overlay->sliderFloat("StepFallOff 10^-2", &StepFallOffMult, 0.0f, 1.0f);

		overlay->sliderFloat("LightStepSize", &VolumetricsData.LightMarchSize, 0.01f, 0.2f);
		overlay->sliderFloat("AbsorptionCoefficient", &VolumetricsData.Absorption, 0.01f, 1.f);
		overlay->sliderFloat("Max Density", &VolumetricsData.Density, 0.01f, 1.f);
		overlay->sliderFloat("AbsorptionCutoff", &VolumetricsData.AbsorptionCutoff, 0.f, 1.f);
		overlay->sliderFloat("LightAbsorptionCutoff", &VolumetricsData.LightAbsorptionCutoff, 0.f, 1.f);

		overlay->sliderFloat("NoiseXTile", &VolumetricsData.NoiseXTile, 0.05f, 40.f);
		overlay->sliderFloat("NoiseYTile", &VolumetricsData.NoiseYTile, 0.05f, 40.f);
		overlay->sliderFloat("NoiseZTile", &VolumetricsData.NoiseZTile, 0.05f, 40.f);
		overlay->sliderFloat("NoiseFactor", &VolumetricsData.NoiseFactor, 0.1f, 15.f);
		overlay->sliderFloat("WindSpeed X", &XWindSpeed, -5.f, 5.f);
		overlay->sliderFloat("WindSpeed Y", &YWindSpeed, -5.f, 5.f);
		overlay->sliderFloat("WindSpeed Z", &ZWindSpeed, -5.f, 5.f);

		overlay->text("\nSDF Data:");
		overlay->sliderFloat("SmoothFactor", &VolumetricsData.SmoothFactor, 0.f, 1.f);
		overlay->sliderInt("SphereIdx", &s_CurrentIMGUISphere, 0, 2);

		overlay->sliderFloat("Sphere X Pos", &FogShapesData.Spheres[s_CurrentIMGUISphere].Pos[0], -20.f, 20.f);
		overlay->sliderFloat("Sphere Y Pos", &FogShapesData.Spheres[s_CurrentIMGUISphere].Pos[1], -20.f, 20.f);
		overlay->sliderFloat("Sphere Z Pos", &FogShapesData.Spheres[s_CurrentIMGUISphere].Pos[2], -20.f, 20.f);

		overlay->sliderFloat("Sphere Radius", &FogShapesData.Spheres[s_CurrentIMGUISphere].Radius, 0.f, 10.f);
		

		SettingsOOD = true;
	}
}