from diffusers import StableDiffusionPipeline

#loading the SD model
pipeline = StableDiffusionPipeline.from_pretrained("face_model_finetuned")
pipeline.to("cpu")  # Use CPU, as mentioned earlier

#generaqting the image with a specific prompt
prompt = "20 year old girl very short black hair green dress looking like a bitch"
generated_image = pipeline(prompt).images[0]


generated_image.save("generated_face.png") #saving annd displaying the img
generated_image.show()
