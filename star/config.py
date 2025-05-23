# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman

import os
path_male_star = os.path.join(os.path.dirname(__file__), '..', 'star_1_1', 'male', 'male.npz')
# 'star_1_1/male'
path_female_star = os.path.join(os.path.dirname(__file__), '..', 'star_1_1', 'female', 'female.npz')
# 'star_1_1/female'
path_neutral_star = os.path.join(os.path.dirname(__file__), '..', 'star_1_1', 'neutral', 'neutral.npz')
# 'star_1_1/neutral'

data_type = 'float32'

if data_type not in ['float16','float32','float64']:
    raise RuntimeError('Invalid data type %s'%(data_type))

class meta(object):
    pass 

cfg = meta()
cfg.data_type = data_type

cfg.path_male_star    = path_male_star
cfg.path_female_star  = path_female_star
cfg.path_neutral_star = path_neutral_star
