import demes
import demesdraw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


def retrieve(estimates, log_opened, repeat, pops, result):
    """
    Retrieve data to eventually compile .csvs
    """
    entries = log_opened[repeat].split('\t') # Retrieve log-likelihood
    ll = entries[1]
    data = [ll]

    for entry in entries: # Retrieve theta
        if 'theta' in entry:
            theta=entry.split("  ")[1].strip(")")
    
    run = entries[0].split(" ")[1] # Retrieve run number for directory access

    """Retrieve parameters from demes model and plot best model"""
    model = demes.load(f"{result}/{run}/final_best_logLL_model_demes_code.py.yml")
    
    tsp_list = [] # Retrieve tsp values
    tsp_record = [] # Make sure not to repeat tsp values
    for population in model.demes:
        if population.name in pops and population.ancestors[0] not in tsp_record:
            ancestor = population.ancestors[0]
            for anc in model.demes:
                if anc.name == ancestor:
                    tsp_list.append(anc.epochs[len((anc).epochs)-1].end_time) #### Check for 3D
                    tsp_record.append(anc.name)
    for tsp in tsp_list:
        data.append(tsp)

    ne_list = [] # Retrieve Ne values
    for pop in pops: #### Will have to accomodate multiple epochs for >2 population models
        for epoch in model[pop].epochs:
            if epoch.end_time == 0:
                try:
                    ne_list.append(epoch.end_size)
                except AttributeError:
                    ne_list.append(epoch.start_size)
    for ne in ne_list:
        data.append(ne)

    mig_list = [0 for i in range(len(mig_list_original))]
    for migration in model.migrations:
        try:
            if migration.start_time: #### Avoid migrations that predate final population split (FOR NOW)
                for mig in range(len(mig_list_original)):
                    if mig_list_original[mig] == f'{migration.source}->{migration.dest}':
                        mig_list[mig] = migration.rate
        except AttributeError:
            pass
    for mig in mig_list:
        data.append(mig)

    """Calculate migrations in terms of migrants per generation"""
    mig_gen_list = []
    for mig in range(len(mig_list)): 
        for ne in range(len(ne_list_original)):
            if ne_list_original[ne].split(' ')[0] == mig_list_original[mig].split('->')[1]:
                
                mig_gen_list.append((mig_list[mig]) * ne_list[ne])

    for mig_gen in mig_gen_list:
        data.append(mig_gen)

    data.append(theta)
    #print(data)

    data.append(run) #### Add run number
    
    estimates.loc[len(estimates)] = data
    #print('Passed')
    return estimates

def parse(log, estimates, csv, demes_model, output, result,repeats):
    """
    Parse GADMA2 log file for parameters
    """
    counter = 0
    for line in reversed(list(open(log))):
        counter+=1
        if line.rstrip() == "Number	log-likelihood	Model	Units":
            best_log = len(open(log).readlines())-counter+1
            for repeat in range(best_log, best_log + repeats): 
                #print(list(open(log))[repeat])
                estimates = retrieve(estimates, list(open(log)), repeat, pops, result)
            break
    estimates.to_csv(csv, index=False)

    """Plot best log-likelihood model""" # Not ideal
    best_model = demes.load(demes_model)
    plot = demesdraw.tubes(best_model, colours = colors_dict)
    plt.savefig(output)




def retrieve_intermediates(result, repeats, intermediates, pops):
    """
    Store intermediate models for plotting, basically modified retrieve function
    """
    storage = pd.DataFrame(columns = ['run', 'intermediates'])
    for run in range(1,repeats+1):
        model_list = []
        for iteration in range(0,10000000000000000000, 100): # Arbitrarily large to check all iterations
            try:
                model = demes.load(f"{result}/{run}/code/demes/iteration_{iteration}.yml")
            except FileNotFoundError:
                break
            data = []
            tsp_list = [] # Retrieve tsp values
            tsp_record = [] # Make sure not to repeat tsp values
            for population in model.demes:
                if population.name in pops and population.ancestors[0] not in tsp_record:
                    ancestor = population.ancestors[0]
                    for anc in model.demes:
                        if anc.name == ancestor:
                            tsp_list.append(anc.epochs[len((anc).epochs)-1].end_time) #### Check for 3D
                            tsp_record.append(anc.name)
            for tsp in tsp_list:
                data.append(tsp)

            ne_list = [] # Retrieve Ne values
            for pop in pops: #### Will have to accomodate multiple epochs for >2 population models
                for epoch in model[pop].epochs:
                    if epoch.end_time == 0:
                        try:
                            ne_list.append(epoch.end_size)
                        except AttributeError:
                            ne_list.append(epoch.start_size)
            for ne in ne_list:
                data.append(ne)

            mig_list = [0 for i in range(len(mig_list_original))]
            for migration in model.migrations:
                try:
                    if migration.start_time: #### Avoid migrations that predate final population split (FOR NOW)
                        for mig in range(len(mig_list_original)):
                            if mig_list_original[mig] == f'{migration.source}->{migration.dest}':
                                mig_list[mig] = migration.rate
                except AttributeError:
                    pass
            for mig in mig_list:
                data.append(mig)

            """Calculate migrations in terms of migrants per generation"""
            mig_gen_list = []
            for mig in range(len(mig_list)): 
                for ne in range(len(ne_list_original)):
                    if ne_list_original[ne].split(' ')[0] == mig_list_original[mig].split('->')[1]:
                        mig_gen_list.append((mig_list[mig]) * ne_list[ne])

            for mig_gen in mig_gen_list:
                data.append(mig_gen)
            model_list.append(data)
        storage.loc[len(storage)] = {'run': run, 'intermediates': model_list}

    storage.to_csv(intermediates, index=False)



if __name__ == "__main__":
    try:
        from snakemake.script import snakemake
        result = snakemake.input['result']
        result_masked = snakemake.input['result_masked']
        best_demes= snakemake.input['best_demes']
        prefix = snakemake.params['prefix']
        pops = snakemake.params['pops'].split('/')
        refGenome = snakemake.params['refGenome']
        gadma_log = snakemake.input['gadma_log']
        best_demes_masked = snakemake.input['best_demes_masked']
        gadma_log_masked = snakemake.input['gadma_log_masked']
        analysis = snakemake.params['analysis']
        spec = snakemake.params['spec']
        colors = ['rebeccapurple', 'steelblue', 'seagreen']
        colors_dict = {key: value for key, value in zip(pops, colors)}
        repeats = snakemake.params['repeats']
        intermediates = snakemake.output['intermediate_models']
        intermediates_masked = snakemake.output['intermediate_models_masked']

    except ImportError:
        parse = argparse.ArgumentParser()
        parse.add_argument('--species_data', '-s',type=str,action='store',help='compiled species results')
        parse.add_argument('--bestLL_demes', '-b',type=str,action='store',help='best LL demes .yml file')
        parse.add_argument('--outfile', '-o',type=str,action='store',help='output file')
        parse.add_argument('--populations', '-ps',type=str,action='store',help='population names')
        parse.add_argument('--refGenome', '-rg',type=str,action='store',help='reference genome')
        args = parse.parse_args()
        bestLL_demes = args.bestLL_demes
        species_data = args.species_data
        outfile = args.outfile
        pops = args.populations
        refGenome = args.refGenome

    """Create empty dataframe to store top five parameter estimates based on LL"""
    cols = ['log-likelihood']
    tsp_list = [f'tsp{i+1} (years)' for i in (range(len(pops)-1))]
    for tsp in tsp_list:
        cols.append(tsp)

    ne_list_original = [f'{pops[i]} Ne' for i in range(len(pops))]
    for ne in ne_list_original:
        cols.append(ne)

    mig_list_original = []
    for pop in pops:
        for other_pop in pops:
            if pop != other_pop:
                mig_list_original.append(f'{pop}->{other_pop}') #### Avoid migrations that predate final population split (FOR NOW)
    for mig in mig_list_original:
        cols.append(mig)

    mig_list_gen = mig_list_original
    for mig in mig_list_gen:
        mig += " per gen"
        cols.append(mig)
    
    cols.append('Theta')

    cols.append('Run')

    estimates = pd.DataFrame(columns = cols)

    """Recreate empty dataframe to store top five masked parameter estimates based on LL"""
    estimates_masked = pd.DataFrame(columns = cols)
    
    """Save intermediate models"""
    retrieve_intermediates(result, repeats, intermediates, pops)
    retrieve_intermediates(result_masked, repeats, intermediates_masked, pops)
    
    """Parse parameters"""
    parse(gadma_log, estimates, snakemake.output['estimates'], best_demes, snakemake.output['best_plot'], snakemake.input['result'], repeats)
    parse(gadma_log_masked, estimates_masked, snakemake.output['estimates_masked'],best_demes_masked, snakemake.output['best_plot_masked'], snakemake.input['result_masked'], repeats)