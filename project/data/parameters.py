# https://bionumbers.hms.harvard.edu/files/Growth%20parameters%20of%20E.%20coli%20growing%20on%20different%20carbon%20sources.pdf
# https://bionumbers.hms.harvard.edu/bionumber.aspx?id=104089&ver=7&trm=glucose+diffusion&org=
# https://pmc.ncbi.nlm.nih.gov/articles/PMC3699654/
grid_params = {
    'length': 50,
    'size_factor': 2,
    'dt': 1/3600,
    'density_diffusion_factor': 20,
}
cell_params = {
    'm_init': 203 * 10**-9, # ug(Xdw)
    'mu_max': 0.89,  # h-1
    'Ks': 7.5 * 10**-14,  # ug um-3
    'v_max': 25 * 3600,  # um h-1 == 3600 square(x) h-1
    't_burst': 0.5,  # h-1
    'dx': 25}  # um == 1 square
substrate_params = {
    'Mw': 180.156,  # g mol-1
    'Y_xs': 11.8*10**-1 / 180.156,  # g(Xdw) g(S)-1
    'D': 673 * 3600,  # um^2 h-1
    'r_in': 0*900 * 10**-9,  # ug um-3 h-1
    'c_init': 2_000 * 10**-9,  # ug um-3  
    'c_max': 2_000 * 10**-9}  # ug um-3  
phage_params = {
    'D': 2.88 * 3600 * grid_params['density_diffusion_factor'],  # um^2 h-1
    'n_burst': 100.0, # PFU
    'delta': 10**-12 * grid_params['dt'],  # um^3 PFU-1
    'r_decay': 0.2,  # h-1
    'c_init': 0.000001,  # PFU um-3
    'c_max': 200 / grid_params['size_factor']**2  # PFU um-3  
}
quorum_params = {
    'D': 800 * 3600,  # um^2 h-1   QUORUM MOLECULES ARE ROUGHLY SAME SIZE AS GLUCOSE, SO SIMILAR DIFFUSION COEF
    'r_production': 90 * 10**-9,  # ug um-3 h-1   SURROUNDING HEALTHY CELLS ARE HELPING
    't_production_factor': 0.2,
    'r_decay': 0,
    'c_init': 0.0001 * 10 **-9,  # ug um-3
    'c_thresh': 0.001 * 10 **-9,  # ug um-3
    'c_max': 0.05 * 10**-9  # ug um-3
}