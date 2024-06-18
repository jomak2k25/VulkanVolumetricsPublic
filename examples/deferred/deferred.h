#pragma once
#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

class VulkanVolumetrics;

class VulkanExample : public VulkanExampleBase
{
	friend VulkanVolumetrics;
public:
	int32_t debugDisplayTarget = 0;

	struct {
		struct {
			vks::Texture2D colorMap;
			vks::Texture2D normalMap;
		} model;
		struct {
			vks::Texture2D colorMap;
			vks::Texture2D normalMap;
		} floor;
	} textures;

	struct {
		vkglTF::Model model;
		vkglTF::Model floor;
	} models;

	struct UniformDataOffscreen {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;
		glm::vec4 instancePos[3];
	} uniformDataOffscreen;

	struct Light {
		glm::vec4 position;
		glm::vec3 color;
		float radius;
	};

	struct UniformDataComposition {
		Light lights[6];
		glm::vec4 viewPos;
		int debugDisplayTarget = 0;
		int lightCount = 3;
	} uniformDataComposition;

	struct {
		vks::Buffer offscreen{ VK_NULL_HANDLE };
		vks::Buffer composition{ VK_NULL_HANDLE };
	} uniformBuffers;

	struct {
		VkPipeline offscreen{ VK_NULL_HANDLE };
		VkPipeline composition{ VK_NULL_HANDLE };
	} pipelines;
	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };

	struct {
		VkDescriptorSet model{ VK_NULL_HANDLE };
		VkDescriptorSet floor{ VK_NULL_HANDLE };
		VkDescriptorSet composition{ VK_NULL_HANDLE };
	} descriptorSets;

	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };

	// Framebuffers holding the deferred attachments
	struct FrameBufferAttachment {
		VkImage image;
		VkDeviceMemory mem;
		VkImageView view;
		VkFormat format;
	};
	struct FrameBuffer {
		int32_t width, height;
		VkFramebuffer frameBuffer;
		// One attachment for every component required for a deferred rendering setup
		FrameBufferAttachment position, normal, albedo;
		FrameBufferAttachment depth;
		VkRenderPass renderPass;
	} offScreenFrameBuf{};

	// One sampler for the frame buffer color attachments
	VkSampler colorSampler{ VK_NULL_HANDLE };

	VkCommandBuffer offScreenCmdBuffer{ VK_NULL_HANDLE };

	// Semaphore used to synchronize between offscreen and final scene rendering
	VkSemaphore offscreenSemaphore{ VK_NULL_HANDLE };

	// Volumetrics to manage the volumetric fog added to the scene
	VulkanVolumetrics Volumetrics;


	VulkanExample();

	~VulkanExample();

	// Enable physical device features required for this example
	virtual void getEnabledFeatures();

	// Create a frame buffer attachment
	void createAttachment(
		VkFormat format,
		VkImageUsageFlagBits usage,
		FrameBufferAttachment* attachment);

	// Prepare a new framebuffer and attachments for offscreen rendering (G-Buffer)
	void prepareOffscreenFramebuffer();

	// Build command buffer for rendering the scene to the offscreen frame buffer attachments
	void buildDeferredCommandBuffer();

	void loadAssets();

	void buildCommandBuffers();

	void setupDescriptors();

	void preparePipelines();

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers();

	// Update matrices used for the offscreen rendering of the scene
	void updateUniformBufferOffscreen();

	// Update lights and parameters passed to the composition shaders
	void updateUniformBufferComposition();

	void prepare();

	void draw();

	virtual void render();

	virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay);
};