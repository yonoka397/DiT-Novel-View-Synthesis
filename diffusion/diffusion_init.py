# Modified from OpenAI's diffusion repository
#     https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
#     https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/losses.py

from diffusion import gaussian_diffusion as gd


def create_diffusion(
    diffusion_steps=1000,
):
    betas = gd.get_named_beta_schedule(diffusion_steps)

    return gd.GaussianDiffusion(
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED_RANGE,
        loss_type=gd.LossType.MSE
    )
