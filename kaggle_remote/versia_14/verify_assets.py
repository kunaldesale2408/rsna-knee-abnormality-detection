import json

datasets = {
    'goodpjw2008/rsna-knee-2-5d-convnext-reader': [
        'cnxt_v0_fold0.pt', 'cnxt_v0_fold1.pt', 'cnxt_v0_fold2.pt', 'infer.py', 'knee.py', 'make_labels.py', 'oof.py', 'preprocess.py', 'pylibjpeg-2.1.0-py3-none-any.whl', 'pylibjpeg_libjpeg-2.4.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl', 'pylibjpeg_libjpeg-2.4.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl', 'pylibjpeg_openjpeg-2.6.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl', 'pylibjpeg_openjpeg-2.6.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl', 'train.py'
    ],
    'heliosli/rsna-knee-compact-r7cnxfsss': [
        'E_lrprior_s0_f0.pt', 'E_lrprior_s0_f1.pt', 'E_lrprior_s0_f2.pt', 'E_lrprior_s0_f3.pt', 'fluid_stacker.json', 'manifest.json', 'side_cls.pt', 'state_spec.json', 'state_stacker.json'
    ],
    'rostigor/rsna-knee-r6-fe055406': [
        'E_lrprior_s0_f0.pt', 'E_lrprior_s0_f1.pt', 'E_lrprior_s0_f2.pt', 'E_lrprior_s0_f3.pt', 'fluid_stacker.json', 'manifest.json', 'side_cls.pt', 'state_spec.json', 'state_stacker.json'
    ],
    'nartaa/rsna-knee-publication-swa-weights-20261007': [
        'README.md', 'checkpoint-manifest.json', 'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt', 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
    ],
    'heliosli/rsna-orthofoundation-fold0-epoch7-20261004': [
        'anatomy.pt'
    ]
}

# 1. Raptor 384 looks for raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
assert 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt' in datasets['nartaa/rsna-knee-publication-swa-weights-20261007']
print("CHECK 1: Raptor 384 weight present: YES")

# 2. Raptor 224 looks for raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt
assert 'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt' in datasets['nartaa/rsna-knee-publication-swa-weights-20261007']
print("CHECK 2: Raptor 224 weight present: YES")

# 3. Stage 2 (R6) looks for C_SLUG 'rsna-knee-r6-fe055406'
assert 'manifest.json' in datasets['rostigor/rsna-knee-r6-fe055406']
print("CHECK 3: R6 manifest present: YES")

# 4. Stage 2b (R7) looks for C_SLUG 'rsna-knee-compact-r7cnxfsss'
assert 'manifest.json' in datasets['heliosli/rsna-knee-compact-r7cnxfsss']
print("CHECK 4: R7 manifest present: YES")

# 5. Part B looks for knee.py and cnxt_v0_fold[012].pt
assert 'knee.py' in datasets['goodpjw2008/rsna-knee-2-5d-convnext-reader']
assert 'cnxt_v0_fold0.pt' in datasets['goodpjw2008/rsna-knee-2-5d-convnext-reader']
print("CHECK 5: 2.5D ConvNeXt reader checkpoints present: YES")
