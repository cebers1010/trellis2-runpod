import runpod
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from PIL import Image
import base64
import io
import tempfile
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
import o_voxel

# Initialize the pipeline globally
try:
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None

def base64_to_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

def image_to_base64_str(image, format="PNG"):
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def handler(job):
    job_input = job['input']
    
    if pipeline is None:
        return {"error": "Model failed to load."}

    # Extract inputs with defaults matching app.py
    try:
        image_input = job_input.get('image')
        if not image_input:
            return {"error": "No image provided in input."}
        
        image = base64_to_image(image_input)
        
        seed = job_input.get('seed', 0)
        resolution = job_input.get('resolution', "1024") # "512", "1024", "1536"
        ss_guidance_strength = job_input.get('ss_guidance_strength', 7.5)
        ss_guidance_rescale = job_input.get('ss_guidance_rescale', 0.7)
        ss_sampling_steps = job_input.get('ss_sampling_steps', 12)
        ss_rescale_t = job_input.get('ss_rescale_t', 5.0)
        
        shape_slat_guidance_strength = job_input.get('shape_slat_guidance_strength', 7.5)
        shape_slat_guidance_rescale = job_input.get('shape_slat_guidance_rescale', 0.5)
        shape_slat_sampling_steps = job_input.get('shape_slat_sampling_steps', 12)
        shape_slat_rescale_t = job_input.get('shape_slat_rescale_t', 3.0)
        
        tex_slat_guidance_strength = job_input.get('tex_slat_guidance_strength', 1.0)
        tex_slat_guidance_rescale = job_input.get('tex_slat_guidance_rescale', 0.0)
        tex_slat_sampling_steps = job_input.get('tex_slat_sampling_steps', 12)
        tex_slat_rescale_t = job_input.get('tex_slat_rescale_t', 3.0)
        
        decimation_target = job_input.get('decimation_target', 500000)
        texture_size = job_input.get('texture_size', 2048)

        # Preprocess Image
        processed_image = pipeline.preprocess_image(image)

        # Run Pipeline
        outputs, latents = pipeline.run(
            processed_image,
            seed=seed,
            preprocess_image=False, # Already preprocessed
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_slat_sampling_steps,
                "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale,
                "rescale_t": shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_slat_sampling_steps,
                "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale,
                "rescale_t": tex_slat_rescale_t,
            },
            pipeline_type={
                "512": "512",
                "1024": "1024_cascade",
                "1536": "1536_cascade",
            }[resolution],
            return_latent=True,
        )

        # Decode Logic (Extract GLB)
        shape_slat, tex_slat, res = latents
        mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
        
        # Save GLB to buffer/temp
        with tempfile.TemporaryDirectory() as tmp_dir:
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=pipeline.pbr_attr_layout,
                grid_size=res,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                use_tqdm=False,
            )
            
            glb_path = os.path.join(tmp_dir, 'output.glb')
            glb.export(glb_path, extension_webp=True)
            
            with open(glb_path, "rb") as f:
                glb_bytes = f.read()
                
            glb_base64 = base64.b64encode(glb_bytes).decode('utf-8')
            
            # Clear CUDA cache
            torch.cuda.empty_cache()

            return {"glb": glb_base64}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
