#pragma once

// Timer Util Class
#include "Timer.h"

#include "VulkanDevice.h"
#include "VulkanTexture.h"
#include "VulkanBuffer.h"
#include "VulkanUIOverlay.h"
#include <array>
#include "glm/common.hpp"

class VulkanExample;
class Camera;

struct SphereInfo
{
	// Centre position of the sphere
	glm::vec3 Pos;
	// Radius of the sphere
	float Radius;
};

struct FogShapes
{
	SphereInfo Spheres[3];
	int SphereCount = 3;
};

struct VolumetricsInfo
{
	// The Albedo of the volumetric fog
	glm::vec3 Albedo;
	// The interval at which the ray marching will sample the fog for opaque visibility
	glm::f32 InitialStepSize;
	// The length the step sizes increase for each step
	glm::f32 StepFallOff;
	// The interval at which the ray marching will sample the fog for light visibility
	glm::f32 LightMarchSize;
	// Clipping distances used to limit the length of the ray marching from the camera
	glm::f32 Near;
	glm::f32 Far;
	// The absorbtion coefficient used to alter how much light is absorbed by the fog
	glm::f32 Absorption;
	// The constant density of fog
	glm::f32 Density;
	// Absorption Cutoff defines how much the opaque and lighting visibility
	// can be reduced before ray marching is halted.
	glm::f32 AbsorptionCutoff;
	glm::f32 LightAbsorptionCutoff;
	// Noise Tiling Frequency
	glm::f32 NoiseXTile;
	glm::f32 NoiseYTile;
	glm::f32 NoiseZTile;
	glm::f32 NoiseXOffset;
	glm::f32 NoiseYOffset;
	glm::f32 NoiseZOffset;
	glm::f32 NoiseFactor;
	// SDF Smooth Union Factor
	glm::f32 SmoothFactor;
	// Max number of steps the ray marching can take
	glm::u32 MapHeight;
	glm::u32 MapWidth;
	glm::u32 MapDepth;
};

struct ComputePipelineResources
{
	VkCommandBuffer CmdBuff{ VK_NULL_HANDLE };				// Command buffer storing the dispatch commands and barriers
	VkSemaphore Semaphore{ VK_NULL_HANDLE };						// Execution dependency betweensubmission
	VkDescriptorSetLayout DescSetLayout{ VK_NULL_HANDLE };	// shader binding layout
	VkDescriptorSet DescSet{ VK_NULL_HANDLE };				// shader bindings
	VkPipelineLayout PipelineLayout{ VK_NULL_HANDLE };				// Layout of the  pipeline
	VkPipeline Pipeline{ VK_NULL_HANDLE };							// Pipeline object
};

class VulkanVolumetrics
{
private:

	void PrepareTextures();

	void PrepareDescriptors();

	void PreparePipelines();

	void BuildCommandBuffers();


public:
	void Init(VulkanExample* example, vks::VulkanDevice* device, Camera* camera, VkQueue* pQueue);

	void UpdateBuffers();

	// Call to submit the volumetrics commands to the GPU. 
	// IN: A semaphore the the submission should wait on.
	// OUT: A semaphore which will be signalled when the Volumetrics have completed
	VkSemaphore* SubmitCommands(VkSemaphore* pWaitSemaphore = VK_NULL_HANDLE);

	void Release(VkDevice& device);

	void UpdateOverlay(vks::UIOverlay* overlay);

	// Store the resources needed for the volumetrics compute commands
	VkQueue ComputeQueue{ VK_NULL_HANDLE };
	VkCommandPool ComputeCmdPool{ VK_NULL_HANDLE };

	std::array<ComputePipelineResources, 2> ComputePipelines;

	// Texture to store the output of the first stage volumetrics compute shader
	vks::Texture FirstStageTexture;

	// Texture to store the output of the second stage volumetrics compute shader
	vks::Texture SecondStageTexture;

	FogShapes FogShapesData;

	VolumetricsInfo VolumetricsData;

	glm::f32 StepFallOffMult = 0.025f;

	vks::Buffer FogShapesBuff;

	vks::Buffer VolumetricsBuff;

	vks::Texture2D PerlinNoise;

	static constexpr int s_NumResources = 1;
	static constexpr int s_VolumeDescSetID = 2;
	static constexpr int s_SphereVolumeBindingID = 0;

	// Flag to check if the volumetrics settings have become out of date
	bool SettingsOOD = false;

	// Vulkan Specific Resources

	VkDescriptorPool DescPool = VK_NULL_HANDLE;

	// Resources for the final descriptor set for use in the main lighting pass
	VkDescriptorSetLayout LightingPassDescSetLayout = VK_NULL_HANDLE;
	VkDescriptorSet LightingPassDescSet = VK_NULL_HANDLE;

	VkPipelineStageFlags SubmitStageFlag = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	VkSubmitInfo SubmitInfo;
	// Base Resources

	vks::VulkanDevice* pDevice;

	VulkanExample* pExampleBase = nullptr;

	Camera* pCamera;

	Timer<resolutions::milliseconds> FrameTimer;

	float XWindSpeed;
	float YWindSpeed;
	float ZWindSpeed;
};