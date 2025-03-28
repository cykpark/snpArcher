import pandas as pd
import argparse

def pc_filter(bounds):
        """
        Sample individuals by PCA grouping
        """
        pop = myVecs[myVecs['PC1'].between(float(bounds[0]),float(bounds[1]))] 
        pop = pop[pop['PC2'].between(float(bounds[2]),float(bounds[3]))]
        pop = pop['#IID']
        return pop
        
def coord_filter(bounds,coords, i):
    """
    Sample individuals by coordinates
    """
    if bounds == 'None' or bounds == 'none':
        return coords['IID']
    else:
        bounds_temp = bounds.split('/')[i].split(',')
        pop_coords = coords[coords['Longitude'] > float(bounds_temp[0])]
        pop_coords = pop_coords[pop_coords['Longitude'] < float(bounds_temp[1])]
        pop_coords = pop_coords[pop_coords['Latitude'] > float(bounds_temp[2])]
        pop_coords = pop_coords[pop_coords['Latitude'] < float(bounds_temp[3])]
        pop_coords = pop_coords['#IID']
        return pop_coords

def create_popfile(pop,label):
    """
    Write to popfile
    """
    pop['Pop'] = label
    with open(f'results/{refGenome}/{analysis}_analysis{spec}/{prefix}.popfile.txt', 'a') as file:
        final_pop = pop.to_string(header=False, index=False).replace(' ', '').replace('{0}'.format(label), ' {0}'.format(label))
        file.write(final_pop + '\n')
        file.close()
        
    with open(f'results/{refGenome}/{analysis}_analysis{spec}/{prefix}.popfile_mixed.txt', 'a') as file:
        final_pop = (pop.iloc[:, 0]).to_string(header=False, index=False).replace(' ', '')
        file.write(final_pop + '\n')
        file.close()


if __name__ == "__main__":
    try:
        from snakemake.script import snakemake
        eigenvec_input = snakemake.input['eigenvec']
        coords_input = snakemake.input['coords']
        prefix = snakemake.params['prefix']
        analysis = snakemake.params['analysis']
        pops = snakemake.params['pops']
        pc_bounds = snakemake.params['pc_bounds']
        coord_bounds = snakemake.params['coord_bounds']
        refGenome = snakemake.params['refGenome']
        spec = snakemake.params['spec']
         
    except ImportError:
        parse = argparse.ArgumentParser()
        parse.add_argument('--vec', '-v',type=str,action='store',help='eigenvec input file')
        parse.add_argument('--coords', '-c',type=str,action='store',help='coordinates input file')
        parse.add_argument('--outfile', '-o',type=str,action='store',help='output file')
        parse.add_argument('--populations', '-ps',type=str,action='store',help='population names')
        parse.add_argument('--pc_bounds', '-pb',type=str,action='store',help='pc bounds')
        parse.add_argument('--coord_bounds', '-cb',type=str,action='store',help='coordinate bounds')
        parse.add_argument('--refGenome', '-rg',type=str,action='store',help='reference genome')
        args = parse.parse_args()
        eigenvec_input = args.vec
        coords_input = args.coords
        out = args.outfile
        pops = args.populations
        pc_bounds = args.pc_bounds
        coord_bounds = args.coord_bounds
        refGenome = args.refGenome

    """
    Read in species eigenvec and coordinate files
    """
    if eigenvec_input != []:
        myVecs = pd.read_table(eigenvec_input)
    pops = pops.split('/')
    if coords_input != []:
        coords_file = pd.read_table(coords_input)
        coords_file.columns = ('#IID', 'Latitude', 'Longitude')

    """
    Sample individuals and write to popfile
    """
    for i in range(len(pops)):
        try:
            pop = pc_filter(pc_bounds.split('/')[i].split(','))
            pop_coords = coord_filter(coord_bounds, coords_file, i)
            pop = pd.merge(pop_coords, pop)
        except AttributeError:
            pop = pd.read_csv(f"results/{refGenome}/pop_analysis/{prefix}_populations/population_{i+1}.txt", sep=" ", header=None)
        create_popfile(pop, pops[i])