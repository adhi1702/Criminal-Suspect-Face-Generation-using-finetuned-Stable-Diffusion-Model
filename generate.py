from diffusers import StableDiffusionPipeline

#loading the SD model
pipeline = StableDiffusionPipeline.from_pretrained("face_model_finetuned")
pipeline.to("cpu")  # Use CPU, as mentioned earlier

#generaqting the image with a specific prompt
prompt = "A female suspect with a 35-40-year-old face, medium build, long blonde hair, green eyes."
generated_image = pipeline(prompt).images[0]


generated_image.save("generated_face.png") #saving annd displaying the img
generated_image.show()
