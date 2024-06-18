@echo off

for /R "shaders\glsl\deferred" %%f in (.) do (
	pushd %%f
		for %%i in (*) do (
			if "%%~xi" neq ".spv" (
				echo found shader %%i
				%VULKAN_SDK%/bin/glslangValidator.exe -V %%i -o "%%i.spv"
			)
		)
	popd
)
pause