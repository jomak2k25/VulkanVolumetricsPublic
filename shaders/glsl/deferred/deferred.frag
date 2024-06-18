#version 450

layout (binding = 1) uniform sampler2D samplerposition;
layout (binding = 2) uniform sampler2D samplerNormal;
layout (binding = 3) uniform sampler2D samplerAlbedo;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragcolor;

struct Light {
	vec4 position;
	vec3 color;
	float radius;
};

layout (binding = 4) uniform UBO 
{
	Light lights[6];
	vec4 viewPos;
	int displayDebugTarget;
	int lightCount;
} ubo;


layout (set = 1, binding = 0, rgba8) uniform readonly image2D FogImage;

void main() 
{
	// Get G-Buffer values
	vec3 fragPos = texture(samplerposition, inUV).rgb;
	vec3 normal = texture(samplerNormal, inUV).rgb;
	vec4 albedo = texture(samplerAlbedo, inUV);
	
	// Debug display
	if (ubo.displayDebugTarget > 0) {
		switch (ubo.displayDebugTarget) {
			case 1: 
				outFragcolor.rgb = fragPos;
				break;
			case 2: 
				outFragcolor.rgb = normal;
				break;
			case 3: 
				outFragcolor.rgb = albedo.rgb;
				break;
			case 4: 
				outFragcolor.rgb = albedo.aaa;
				break;
		}		
		outFragcolor.a = 1.0;
		return;
	}

	// Render-target composition
	#define ambient 0.0
	
	// Ambient part
	vec3 fragcolor  = albedo.rgb * ambient;
	
	for(int i = 0; i < ubo.lightCount; ++i)
	{
		// Vector to light
		vec3 L = ubo.lights[i].position.xyz - fragPos;
		// Distance from light to fragment position
		float dist = length(L);

		// Viewer to fragment
		vec3 V = ubo.viewPos.xyz - fragPos;
		V = normalize(V);
		
		//if(dist < ubo.lights[i].radius)
		{
			// Light to fragment
			L = normalize(L);

			// Attenuation
			float atten = ubo.lights[i].radius / (pow(dist, 2.0) + 1.0);

			// Diffuse part
			vec3 N = normalize(normal);
			float NdotL = max(0.0, dot(N, L));
			vec3 diff = ubo.lights[i].color * albedo.rgb * NdotL * atten;

			// Specular part
			// Specular map values are stored in alpha of albedo mrt
			vec3 R = reflect(-L, N);
			float NdotR = max(0.0, dot(R, V));
			vec3 spec = ubo.lights[i].color * albedo.a * pow(NdotR, 16.0) * atten;

			fragcolor += diff + spec;	
		}	
	}    	
   
  // Apply the volumetric fog to the final image by blending with the fog texture
  ivec2 FogImageSize = imageSize(FogImage);
  vec4 FogColour = imageLoad(FogImage, ivec2(inUV.x * FogImageSize.x, inUV.y * FogImageSize.y));
  fragcolor = mix(fragcolor, FogColour.xyz, FogColour.w);

  outFragcolor = vec4(fragcolor, 1.0);	
}