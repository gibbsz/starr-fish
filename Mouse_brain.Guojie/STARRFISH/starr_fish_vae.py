from scvi.module._multivae import MULTIVAE, mix_modalities, auto_move_data, REGISTRY_KEYS, \
    get_reconstruction_loss_protein, kld, LossOutput, NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.model._multivi import MULTIVI
import torch
from torch.nn import functional as F
from scvi.module._peakvae import Decoder as DecoderPeakVI
from scvi.nn._base_components import DecoderSCVI
from torch.distributions import Normal, Poisson
from scvi._types import AnnOrMuData
from collections.abc import Sequence
from typing import Literal, Iterable
import numpy as np
import pandas as pd
from mudata import MuData
from scipy.sparse import csr_matrix, vstack
from scvi.utils._docstrings import de_dsp
from anndata import AnnData
from functools import partial
from scvi.model.base._de_core import _de_core
from scvi.model._utils import (
    _get_batch_code_from_category,
    scrna_raw_counts_properties
)


class DecoderInfectionRateVI(DecoderPeakVI):
    '''
    Decoder for AAV infection rate, get latent representation, decode infection rate, use softplus activation
    It should follow a poisson distribution: https://pubmed.ncbi.nlm.nih.gov/20703310/
    '''
    def __init__(self, 
                 n_input: int,
                 n_output: int,
                 n_cat_list: Iterable[int] = None,
                 n_layers: int = 2,
                 n_hidden: int = 128,
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = True,
                 deep_inject_covariates: bool = False,
                 **kwargs,):
        super().__init__(n_input=n_input, n_output=n_output, n_cat_list=n_cat_list, n_layers=n_layers, n_hidden=n_hidden,
                         use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm, deep_inject_covariates=deep_inject_covariates,
                         **kwargs)
        self.output = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_output), torch.nn.Softplus())


class STARRFISHVAE(MULTIVAE):
    def __init__(
        self, r_max: int = 10, infection_rate_prior: float = None,
        n_input_regions: int = 0,
        n_input_genes: int = 0,
        n_input_proteins: int = 0,
        modality_weights: Literal["equal", "cell", "universal"] = "equal",
        modality_penalty: Literal["Jeffreys", "MMD", "None"] = "Jeffreys",
        n_batch: int = 0,
        n_obs: int = 0,
        n_labels: int = 0,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        gene_dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        n_hidden: int = None,
        n_latent: int = None,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate: float = 0.1,
        region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: Literal["normal", "ln"] = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
        use_size_factor_key: bool = False,
        protein_background_prior_mean: np.ndarray | None = None,
        protein_background_prior_scale: np.ndarray | None = None,
        protein_dispersion: str = "protein",
        infection_rate_inference: str = "encoder",
        infection_rate_generative: str = "sample",
        infection_rate_type: Literal["gene", "gene-cell"] = "gene",
        accessibility_generative: str = "split",
        kl_infection_rate_type: Literal["gene-multinomial", "global-poisson", "gene-multinomial+global-poisson", "gene-cosine", "gene-cosine+global-poisson"] = "gene-cosine",
        infection_rate_library_size = None,
    ):
        if infection_rate_prior is None:
            infection_rate_prior = 0.08
        super().__init__(n_input_regions=n_input_regions, n_input_genes=n_input_genes, n_input_proteins=n_input_proteins,
                         modality_weights=modality_weights, modality_penalty=modality_penalty, n_batch=n_batch, n_obs=n_obs,
                         n_labels=n_labels, gene_likelihood=gene_likelihood, gene_dispersion=gene_dispersion, n_hidden=n_hidden,
                         n_latent=n_latent, n_layers_encoder=n_layers_encoder, n_layers_decoder=n_layers_decoder,
                         n_continuous_cov=n_continuous_cov, n_cats_per_cov=n_cats_per_cov, dropout_rate=dropout_rate,
                         region_factors=region_factors, use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm,
                         latent_distribution=latent_distribution, deeply_inject_covariates=deeply_inject_covariates,
                         encode_covariates=encode_covariates, use_size_factor_key=use_size_factor_key,
                         protein_background_prior_mean=protein_background_prior_mean, protein_background_prior_scale=protein_background_prior_scale,
                         protein_dispersion=protein_dispersion)
        self.r_max = r_max
        self.infection_rate_prior = infection_rate_prior
        cat_list = [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else []
        self.reporter_dispersion = "gene"
        self.pa_r = torch.nn.Parameter(torch.randn(self.n_input_regions))
        # NOTE: 
        # To model the activity of the enhancer, we modify self.z_decoder_accessibility, we can use DecoderSCVI
        # Original accessibility decoder
        # self.z_decoder_accessibility = DecoderPeakVI(
        #     n_input=self.n_latent + self.n_continuous_cov,
        #     n_output=n_input_regions,
        #     n_hidden=self.n_hidden,
        #     n_cat_list=cat_list,
        #     n_layers=self.n_layers_decoder,
        #     use_batch_norm=self.use_batch_norm_decoder,
        #     use_layer_norm=self.use_layer_norm_decoder,
        #     deep_inject_covariates=self.deeply_inject_covariates,
        # )
        self.z_decoder_accessibility = DecoderSCVI(
            n_input=self.n_latent + self.n_continuous_cov,
            n_output=self.n_input_regions,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            n_hidden=self.n_hidden,
            inject_covariates=self.deeply_inject_covariates,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            scale_activation="softplus" if self.use_size_factor_key else "softmax",
        )
        # NOTE:
        # To model the infection rate of AAV, we modify self.l_encoder_accessibility, 
        # There are two strategies
        # 1. we change this to a new encoder, with softplus activation
        # 2. we make it a decoder, change the activation function to softplus, we take this approach for now
        # To avoid confusion, we rename it to self.z_decoder_infection_rate, we can use DecoderPeakVI, change activation to softplus
        # Remove Original accessibility encoder
        self.l_encoder_accessibility = None
        self.accessibility_generative = accessibility_generative
        self.infection_rate_inference = infection_rate_inference
        self.infection_rate_generative = infection_rate_generative
        self.infection_rate_type = infection_rate_type
        self.kl_infection_rate_type = kl_infection_rate_type
        # if kl_infection_rate_type contains "gene-multinomial" or "gene-cosine", we need to add a new parameter
        if "gene-multinomial" in kl_infection_rate_type or "gene-cosine" in kl_infection_rate_type:
            assert infection_rate_library_size is not None, "infection_rate_library_size should be provided"
            if infection_rate_inference == 'encoder':
                assert infection_rate_type == "gene", "infection_rate_type should be gene"
            self.infection_rate_library_size = torch.tensor(infection_rate_library_size, dtype=torch.long).reshape(-1)
        if self.infection_rate_type == "gene-cell":
            n_output_infection_rate = self.n_input_regions
        else:
            n_output_infection_rate = 1
            if type(self.infection_rate_prior) == float:
                self.infection_rate_gene = torch.nn.Parameter(
                    torch.log(torch.exp(self.infection_rate_prior * torch.ones(self.n_input_regions)) - 1), 
                    requires_grad=True
                )
            else:
                self.infection_rate_gene = torch.nn.Parameter(
                    torch.log(torch.exp(self.infection_rate_prior) - 1), 
                    requires_grad=True
                )
        if self.infection_rate_inference == "encoder":
            if self.n_input_genes == 0:
                input_exp = 1
            else:
                input_exp = self.n_input_genes
            n_input_encoder_exp = input_exp + n_continuous_cov * encode_covariates
            self.l_encoder_infection_rate = DecoderInfectionRateVI(
                n_input=n_input_encoder_exp,
                n_output=n_output_infection_rate,
                n_hidden=self.n_hidden,
                n_cat_list=cat_list,
                n_layers=self.n_layers_decoder,
                use_batch_norm=self.use_batch_norm_decoder,
                use_layer_norm=self.use_layer_norm_decoder,
                deep_inject_covariates=self.deeply_inject_covariates,
            )
        else:
            self.z_decoder_infection_rate = DecoderInfectionRateVI(
                n_input=self.n_latent + self.n_continuous_cov,
                n_output=n_output_infection_rate,
                n_hidden=self.n_hidden,
                n_cat_list=cat_list,
                n_layers=self.n_layers_decoder,
                use_batch_norm=self.use_batch_norm_decoder,
                use_layer_norm=self.use_layer_norm_decoder,
                deep_inject_covariates=self.deeply_inject_covariates,
            )
        # set T7 decoder parameters
        self.py_scale = torch.nn.Parameter(torch.zeros(1))
        self.py_r = torch.nn.Parameter(torch.zeros(1))
        self.py_rate = torch.nn.Parameter(torch.zeros(1))
        self.py_dropout = torch.nn.Parameter(torch.zeros(1))

    @auto_move_data
    def inference(
        self,
        x,
        y,
        batch_index,
        cont_covs,
        cat_covs,
        label,
        cell_idx,
        size_factor,
        n_samples=1,
    ) -> dict[str, torch.Tensor]:
        """Run the inference model."""
        # Get Data and Additional Covs
        if self.n_input_genes == 0:
            x_rna = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            x_rna = x[:, : self.n_input_genes]
        if self.n_input_regions == 0:
            x_atac = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            x_atac = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        
        mask_expr = x_rna.sum(dim=1) > 0
        mask_acc = x_atac.sum(dim=1) > 0
        mask_pro = y.sum(dim=1) > 0
        
        if cont_covs is not None and self.encode_covariates:
            encoder_input_expression = torch.cat((x_rna, cont_covs), dim=-1)
            encoder_input_accessibility = torch.cat((x_atac, cont_covs), dim=-1)
            encoder_input_protein = torch.cat((y, cont_covs), dim=-1)
        else:
            encoder_input_expression = x_rna
            encoder_input_accessibility = x_atac
            encoder_input_protein = y

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
            
        # Z Encoders
        qzm_acc, qzv_acc, z_acc = self.z_encoder_accessibility(
            encoder_input_accessibility, batch_index, *categorical_input
        )
        qzm_expr, qzv_expr, z_expr = self.z_encoder_expression(
            encoder_input_expression, batch_index, *categorical_input
        )
        qzm_pro, qzv_pro, z_pro = self.z_encoder_protein(
            encoder_input_protein, batch_index, *categorical_input
        )

        # NOTE: L encoders, modified here, make the libsize_acc same as libsize_expr
        if self.use_size_factor_key:
            libsize_expr = torch.log(size_factor[:, [0]] + 1e-6)
            libsize_acc = size_factor[:, [1]]
        else:
            libsize_expr = self.l_encoder_expression(
                encoder_input_expression, batch_index, *categorical_input
            )
            if self.infection_rate_inference == "encoder":
                libsize_acc = self.l_encoder_infection_rate(
                    encoder_input_expression, batch_index, *categorical_input
                )
                if self.infection_rate_type == "gene":
                    libsize_acc = libsize_acc * torch.nn.functional.softplus(self.infection_rate_gene)
            else:
                libsize_acc = libsize_expr

        # mix representations
        if self.modality_weights == "cell":
            weights = self.mod_weights[cell_idx, :]
        else:
            weights = self.mod_weights.unsqueeze(0).expand(len(cell_idx), -1)

        qz_m = mix_modalities(
            (qzm_expr, qzm_acc, qzm_pro), (mask_expr, mask_acc, mask_pro), weights
        )
        qz_v = mix_modalities(
            (qzv_expr, qzv_acc, qzv_pro),
            (mask_expr, mask_acc, mask_pro),
            weights,
            torch.sqrt,
        )

        # sample
        if n_samples > 1:

            def unsqz(zt, n_s):
                return zt.unsqueeze(0).expand((n_s, zt.size(0), zt.size(1)))

            untran_za = Normal(qzm_acc, qzv_acc.sqrt()).sample((n_samples,))
            z_acc = self.z_encoder_accessibility.z_transformation(untran_za)
            untran_ze = Normal(qzm_expr, qzv_expr.sqrt()).sample((n_samples,))
            z_expr = self.z_encoder_expression.z_transformation(untran_ze)
            untran_zp = Normal(qzm_pro, qzv_pro.sqrt()).sample((n_samples,))
            z_pro = self.z_encoder_protein.z_transformation(untran_zp)

            libsize_expr = unsqz(libsize_expr, n_samples)
            libsize_acc = unsqz(libsize_acc, n_samples)

        # sample from the mixed representation
        untran_z = Normal(qz_m, qz_v.sqrt()).rsample()
        z = self.z_encoder_accessibility.z_transformation(untran_z)
        # sample infection rate from libsize_acc
        libsize_expr = {"libsize_expr": libsize_expr, "libsize_acc": libsize_acc, "z_acc": z_acc}
        if self.infection_rate_inference == 'encoder' and self.infection_rate_generative == "sample":
            libsize_acc = torch.poisson(libsize_acc)
        outputs = {
            "x": x,
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "z_expr": z_expr,
            "qzm_expr": qzm_expr,
            "qzv_expr": qzv_expr,
            "z_acc": z_acc,
            "qzm_acc": qzm_acc,
            "qzv_acc": qzv_acc,
            "z_pro": z_pro,
            "qzm_pro": qzm_pro,
            "qzv_pro": qzv_pro,
            "libsize_expr": libsize_expr,
            "libsize_acc": libsize_acc,
        }
        return outputs
        
    @auto_move_data
    def generative(
        self,
        z,
        qz_m,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        libsize_expr=None,
        use_z_mean=False,
        label: torch.Tensor = None,
    ):
        """Runs the generative model."""
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        latent = z if not use_z_mean else qz_m
        if cont_covs is None:
            decoder_input = latent
        elif latent.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [latent, cont_covs.unsqueeze(0).expand(latent.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([latent, cont_covs], dim=-1)
        if self.accessibility_generative == "split":
            accessibility_decoder_input = libsize_expr['z_acc']
        else:
            accessibility_decoder_input = decoder_input

        # Reporter Decoder, modified here
        if self.infection_rate_inference == "encoder":
            pa_infection_rate = libsize_expr['libsize_acc']
        else:
            pa_infection_rate = self.z_decoder_infection_rate(decoder_input, batch_index, *categorical_input)
            if self.infection_rate_type == "gene":
                pa_infection_rate = pa_infection_rate * torch.nn.functional.softplus(self.infection_rate_gene)

        # NOTE: should have three outputs, pa_scale, pa_rate, pa_infection_rate
        pa_scale, _, pa_rate, pa_dropout = self.z_decoder_accessibility(
            self.gene_dispersion,
            accessibility_decoder_input, 
            libsize_expr['libsize_expr'],
            batch_index, 
            *categorical_input,
            label
        )

        # Expression Decoder
        px_scale, _, px_rate, px_dropout = self.z_decoder_expression(
            self.gene_dispersion,
            decoder_input,
            libsize_expr['libsize_expr'],
            batch_index,
            *categorical_input,
            label,
        )
        # Expression Dispersion
        if self.gene_dispersion == "gene-label":
            px_r = F.linear(
                F.one_hot(label.squeeze(-1), self.n_labels).float(), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        # T7 Decoder, should only have two parameters, py_scale, py_rate, py_dropout, should be independent from the latent expression
        return {
            # reporter activity
            "pa_inf_rate": pa_infection_rate,
            "pa_scale": pa_scale,
            "pa_r": torch.exp(self.pa_r),
            "pa_rate": pa_rate,
            "pa_dropout": pa_dropout,
            # expression
            "px_scale": px_scale,
            "px_r": torch.exp(self.px_r),
            "px_rate": px_rate,
            "px_dropout": px_dropout,
            # protein, not used in the current model
            "py_scale": torch.nn.functional.softplus(self.py_scale),
            'py_r': torch.exp(self.py_r),  # Protein Dispersion
            "py_rate": torch.nn.functional.softplus(self.py_rate),
            "py_dropout": self.py_dropout,
        }
    
    # NOTE: 
    # modified the reconstruction loss for accessibility (now the reporter activity)
    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        """Computes the loss function for the model."""
        # Get the data
        x = inference_outputs["x"]

        x_rna = x[:, : self.n_input_genes]
        x_atac = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        if self.n_input_proteins == 0:
            y = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        mask_expr = x_rna.sum(dim=1) > 0
        mask_acc = x_atac.sum(dim=1) > 0
        mask_t7 = y.sum(dim=1) > 0

        # NOTE: Compute Accessibility loss, modified here
        pa_infection_rate = generative_outputs["pa_inf_rate"]
        if self.infection_rate_generative == "sample":
            pa_infection_rate = torch.poisson(pa_infection_rate)
        pa_scale = generative_outputs["pa_scale"]
        pa_rate = generative_outputs["pa_rate"]
        pa_dropout = generative_outputs["pa_dropout"]
        rl_accessibility = self.get_reconstruction_loss_accessibility(x_atac, pa_infection_rate, pa_rate, pa_scale, pa_dropout)
        
        # compute T7 loss
        if mask_t7.sum().gt(0):
            py_scale = generative_outputs["py_scale"]
            py_rate = generative_outputs["py_rate"]
            py_dropout = generative_outputs["py_dropout"]
            rl_t7 = self.get_reconstruction_loss_accessibility(y, pa_infection_rate, py_rate, py_scale, py_dropout)
        else:
            rl_t7 = torch.zeros(x.shape[0], device=x.device, requires_grad=False)

        # Compute Expression loss
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        x_expression = x[:, : self.n_input_genes]
        rl_expression = self.get_reconstruction_loss_expression(x_expression, px_rate, px_r, px_dropout)

        # calling without weights makes this act like a masked sum
        recon_loss_expression = rl_expression * mask_expr
        recon_loss_accessibility = rl_accessibility * mask_acc
        recon_loss_t7 = rl_t7 * mask_t7
        recon_loss = recon_loss_expression + recon_loss_accessibility + recon_loss_t7

        # Compute KLD between Z and N(0,I)
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        kl_div_z = kld(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(0, 1),
        ).sum(dim=1)

        # Compute KLD between distributions for paired data
        kl_div_paired = self._compute_mod_penalty(
            (inference_outputs["qzm_expr"], inference_outputs["qzv_expr"]),
            (inference_outputs["qzm_acc"], inference_outputs["qzv_acc"]),
            (inference_outputs["qzm_pro"], inference_outputs["qzv_pro"]),
            mask_expr,
            mask_acc,
            mask_t7,
        )

        # split kl_infection_rate_type by "+"
        kl_div_infection_rate = 0
        for i in self.kl_infection_rate_type.split("+"):
            if i not in ["gene-multinomial", "global-poisson", "gene-cosine", '']:
                raise ValueError(f"Invalid kl_infection_rate_type: {i}")
            if i == "global-poisson":
                kl_div_infection_rate += kld(
                    Poisson(generative_outputs["pa_inf_rate"]),
                    Poisson(self.infection_rate_prior),
                ).sum(axis=1)
            elif i == "gene-multinomial":
                if self.infection_rate_inference == 'encoder':
                    probs = torch.nn.functional.softplus(self.infection_rate_gene)
                else:
                    probs = generative_outputs["pa_inf_rate"].mean(dim=0)
                kl_div_infection_rate += -torch.distributions.Multinomial(
                    total_count=self.infection_rate_library_size.sum().item(),
                    probs=probs,
                ).log_prob(self.infection_rate_library_size.to(self.infection_rate_gene.device)).sum()
            elif i == "gene-cosine":
                # cosine similarity 
                lib_size = self.infection_rate_library_size.to(torch.float)
                lib_size_centered = lib_size - lib_size.mean()
                if self.infection_rate_inference == 'encoder':
                    infection_rate = torch.nn.functional.softplus(self.infection_rate_gene)
                else:
                    infection_rate = generative_outputs["pa_inf_rate"].mean(dim=0)
                infection_rate_centered = infection_rate - infection_rate.mean()
                kl_div_infection_rate += -torch.nn.functional.cosine_similarity(
                    lib_size_centered.to(infection_rate_centered.device), infection_rate_centered, dim=0
                ).sum()
        # KL WARMUP
        kl_local_for_warmup = kl_div_z
        weighted_kl_local = kl_weight * (kl_local_for_warmup + kl_div_infection_rate) + kl_div_paired

        # TOTAL LOSS
        loss = torch.mean(recon_loss + weighted_kl_local)

        recon_losses = {
            "reconstruction_loss_expression": recon_loss_expression,
            "reconstruction_loss_accessibility": recon_loss_accessibility,
            "reconstruction_loss_protein": recon_loss_t7,
        }
        kl_local = {
            "kl_divergence_z": kl_div_z + kl_div_infection_rate,
            "kl_divergence_paired": kl_div_paired,
        }
        return LossOutput(loss=loss, reconstruction_loss=recon_losses, kl_local=kl_local)

    def get_reconstruction_loss_accessibility(self, x, infection_rate, rate, r, dropout):
        """Computes the reconstruction loss for the reporter activity data."""
        # mu is porportion to the infected number of AAV, r is the number of infected AAV
        pa_rate_r = rate * infection_rate
        pa_r_r = r * (infection_rate + 1e-64)  # avoid division by zero
        if self.gene_likelihood == "zinb":
            rl = (
                ZeroInflatedNegativeBinomial(mu=pa_rate_r, theta=pa_r_r, zi_logits=dropout)
                .log_prob(x)
            )
        elif self.gene_likelihood == "nb":
            rl = NegativeBinomial(mu=pa_rate_r, theta=pa_r_r).log_prob(x)
        elif self.gene_likelihood == "poisson":
            rl = Poisson(pa_rate_r).log_prob(x)
        else:
            raise NotImplementedError("Invalid gene_likelihood")
        # times up the poisson probability and the likelihood
        rl = -rl.sum(dim=-1)
        return rl

        
class STARRFISHVI(MULTIVI):
    _module_cls = STARRFISHVAE
    def __init__(self, 
                 adata: AnnOrMuData,
                 n_genes: int | None = None,
                 n_regions: int | None = None,
                 modality_weights: Literal["equal", "cell", "universal"] = "equal",
                 modality_penalty: Literal["Jeffreys", "MMD", "None"] = "Jeffreys",
                 n_hidden: int | None = None,
                 n_latent: int | None = None,
                 n_layers_encoder: int = 2,
                 n_layers_decoder: int = 2,
                 dropout_rate: float = 0.1,
                 region_factors: bool = True,
                 gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
                 dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
                 use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
                 use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
                 latent_distribution: Literal["normal", "ln"] = "normal",
                 deeply_inject_covariates: bool = False,
                 encode_covariates: bool = False,
                 fully_paired: bool = False,
                 protein_dispersion: Literal["protein", "protein-batch", "protein-label"] = "protein",
                 **model_kwargs,):
        super().__init__(adata=adata, n_genes=n_genes, n_regions=n_regions, modality_weights=modality_weights, 
                         modality_penalty=modality_penalty, n_hidden=n_hidden, n_latent=n_latent, n_layers_encoder=n_layers_encoder, 
                         n_layers_decoder=n_layers_decoder, dropout_rate=dropout_rate, region_factors=region_factors, gene_likelihood=gene_likelihood, 
                         dispersion=dispersion, use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm, latent_distribution=latent_distribution, 
                         deeply_inject_covariates=deeply_inject_covariates, encode_covariates=encode_covariates, fully_paired=fully_paired, 
                         protein_dispersion=protein_dispersion, **model_kwargs)
        if "n_proteins" in self.summary_stats:
            n_proteins = self.summary_stats.n_proteins
        else:
            n_proteins = 0
        self._model_summary_string = (
            f"STARRFISHVI Model with the following params: \nn_genes: {n_genes}, "
            f"n_regions: {n_regions}, T7: {n_proteins}, n_hidden: {self.module.n_hidden}, "
            f"n_latent: {self.module.n_latent}, n_layers_encoder: {n_layers_encoder}, "
            f"n_layers_decoder: {n_layers_decoder}, dropout_rate: {dropout_rate}, "
            f"latent_distribution: {latent_distribution}, "
            f"deep injection: {deeply_inject_covariates}, gene_likelihood: {gene_likelihood}, "
            f"gene_dispersion:{dispersion}, Mod.Weights: {modality_weights}, "
            f"Mod.Penalty: {modality_penalty}, protein_dispersion: {protein_dispersion}"
        )

    @torch.inference_mode()
    def get_accessibility_estimates(
        self,
        adata: AnnOrMuData | None = None,
        indices: Sequence[int] = None,
        n_samples_overall: int | None = None,
        region_list: Sequence[str] | None = None,
        transform_batch: str | int | None = None,
        use_z_mean: bool = True,
        threshold: float | None = None,
        normalize_cells: bool = False,
        normalize_regions: bool = False,
        batch_size: int = 128,
        return_numpy: bool = False,
        key='pa_scale'
    ) -> np.ndarray | csr_matrix | pd.DataFrame:
        """Impute the full accessibility matrix.

        Returns a matrix of accessibility probabilities for each cell and genomic region in the
        input (for return matrix A, A[i,j] is the probability that region j is accessible in cell
        i).

        Parameters
        ----------
        adata
            AnnOrMuData object that has been registered with scvi. If `None`, defaults to the
            AnnOrMuData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to return in total
        region_list
            Regions to use. if `None`, all regions are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
        use_z_mean
            If True (default), use the distribution mean. Otherwise, sample from the distribution.
        threshold
            If provided, values below the threshold are replaced with 0 and a sparse matrix
            is returned instead. This is recommended for very large matrices. Must be between 0 and
            1.
        normalize_cells
            Whether to reintroduce library size factors to scale the normalized probabilities.
            This makes the estimates closer to the input, but removes the library size correction.
            False by default.
        normalize_regions
            Whether to reintroduce region factors to scale the normalized probabilities. This makes
            the estimates closer to the input, but removes the region-level bias correction. False
            by default.
        batch_size
            Minibatch size for data loading into model
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        post = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        if region_list is None:
            region_mask = slice(None)
        else:
            region_mask = [region in region_list for region in adata.var_names[: self.n_regions]]

        if threshold is not None and (threshold < 0 or threshold > 1):
            raise ValueError("the provided threshold must be between 0 and 1")

        imputed = []
        for tensors in post:
            get_generative_input_kwargs = {"transform_batch": transform_batch[0]}
            generative_kwargs = {"use_z_mean": use_z_mean}
            inference_outputs, generative_outputs = self.module.forward(
                tensors=tensors,
                get_generative_input_kwargs=get_generative_input_kwargs,
                generative_kwargs=generative_kwargs,
                compute_loss=False,
            )
            # NOTE: Modified here
            if ":" in key:
                key = key.split(":")
                p = generative_outputs[key[0]][key[1]].cpu()
            else:
                p = generative_outputs[key].cpu()

            if normalize_cells:
                p *= inference_outputs["libsize_acc"].cpu()
            if normalize_regions:
                p *= torch.sigmoid(self.module.region_factors).cpu()
            if threshold:
                p[p < threshold] = 0
                p = csr_matrix(p.numpy())
            if region_mask is not None:
                p = p[:, region_mask]
            imputed.append(p)

        if threshold:  # imputed is a list of csr_matrix objects
            imputed = vstack(imputed, format="csr")
        else:  # imputed is a list of tensors
            imputed = torch.cat(imputed).numpy()

        if np.all(imputed is None):
            return pd.DataFrame(
                imputed,
                index=adata.obs_names[indices],
                columns=[],
            )
        else:
            if return_numpy:
                return imputed
            elif threshold:
                return pd.DataFrame.sparse.from_spmatrix(
                    imputed,
                    index=adata.obs_names[indices],
                    columns=adata["rna"].var_names[: self.n_regions][region_mask]
                    if isinstance(adata, MuData)
                    else adata.var_names[: self.n_regions][region_mask],
                )
            else:
                return pd.DataFrame(
                    imputed,
                    index=adata.obs_names[indices],
                    columns=adata["rna"].var_names[: self.n_regions][region_mask]
                    if isinstance(adata, MuData)
                    else adata.var_names[self.n_genes : (self.n_genes + self.n_regions)][region_mask],
                )


    @torch.inference_mode()
    def get_infection_rate_estimate(self, **kwargs):
        return self.get_accessibility_estimates(key='pa_inf_rate', **kwargs)    


    @torch.inference_mode()
    def get_library_size_factors(
        self,
        adata: AnnOrMuData | None = None,
        indices: Sequence[int] = None,
        batch_size: int = 128,
    ) -> dict[str, np.ndarray]:
        """Return library size factors.

        Parameters
        ----------
        adata
            AnnOrMuData object with equivalent structure to initial AnnData. If `None`, defaults
            to the AnnOrMuData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        Library size factor for expression and accessibility
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        lib_exp = []
        lib_acc = []
        for tensors in scdl:
            outputs = self.module.inference(**self.module._get_inference_input(tensors))
            lib_exp.append(outputs["libsize_expr"]["libsize_expr"].cpu())
            lib_acc.append(outputs["libsize_expr"]["libsize_acc"].cpu())

        return {
            "expression": torch.cat(lib_exp).numpy().squeeze(),
            "accessibility": torch.cat(lib_acc).numpy().squeeze(),
        }
        
    
    @de_dsp.dedent
    def differential_accessibility(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | None = None,
        idx2: Sequence[int] | Sequence[bool] | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""A unified method for differential expression analysis.

        Implements `"vanilla"` DE :cite:p:`Lopez18` and `"change"` mode DE :cite:p:`Boyeau19`.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential expression DataFrame.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)

        col_names = adata.var_names[self.n_genes: (self.n_genes+self.n_regions)]
        model_fn = partial(
            self.get_accessibility_estimates,
            batch_size=batch_size,
        )
        all_stats_fn = partial(
            scrna_raw_counts_properties,
            var_idx=np.arange(adata.shape[1])[self.n_genes: (self.n_genes+self.n_regions)],
        )
        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=None,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=all_stats_fn,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            pseudocounts=1e-6,
            **kwargs,
        )

        return result
    