:: TODO: The --target-env=vulkan1.0 should maybe be left out (but it's working now so I don't want to touch it)
:: glslc -O --target-env=vulkan1.0 -ocmdApplyTestLineshape.spv cmdApplyTestLineshape.comp
glslc -O --target-env=vulkan1.0 -ocmdTestShader1.spv cmdTestShader1.comp

pause
