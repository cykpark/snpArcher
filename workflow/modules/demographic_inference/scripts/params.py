import pandas as pd
import argparse

if __name__ == "__main__":
    try:
        from snakemake.script import snakemake
        bed = snakemake.input['bed']
        prefix = snakemake.params['prefix']
        analysis = snakemake.params['analysis']
        pops = snakemake.params['pops']
        refGenome = snakemake.params['refGenome']
        mutrate = snakemake.params['mutation_rate']
        gentime = snakemake.params['generation_time']
        spec = snakemake.params['spec']
        repeats = snakemake.params['repeats']

    except ImportError:
        parse = argparse.ArgumentParser()
        parse.add_argument('--bed', '-b',type=str,action='store',help='callable sites BED file')
        parse.add_argument('--prefix', '-p',type=str,action='store',help='species prefix')
        parse.add_argument('--mut_rate', '-mr',type=str,action='store',help='species mutation rate')
        parse.add_argument('--gen_time', '-gt',type=str,action='store',help='species generation time')
        parse.add_argument('--mask', '-m',type=str,action='store',help='masking preference')
        parse.add_argument('--populations', '-ps',type=str,action='store',help='population names')
        parse.add_argument('--refGenome', '-rg',type=str,action='store',help='reference genome')
        args = parse.parse_args()
        bed = args.bed
        prefix = args.prefix
        mutrate = args.mut_rate
        gentime = args.gen_time
        pops = args.populations
        refGenome = args.refGenome

    """Define parameters"""
    processes = '25'
    pts = '70, 80, 90'
    engine = 'moments'

    """Read raw bed file and find total sequence length"""
    df = pd.read_csv(bed, sep='\t', comment='t', header=None)
    header = ['chrom', 'chromStart', 'chromEnd']
    df.columns = header[:len(df.columns)]
    df = df.drop(['chrom'], axis=1) # Remove chromosome names
    diff = df.diff(axis=1) # Take difference of the two columns
    diff = diff.drop(['chromStart'], axis=1) # Remove extraneous column
    seq_len = repr(diff.sum()).split('    ')[1].split('\n')[0] # Convert to printable representation

    """
    Write to params files
    """
    params = str(prefix) + '__' + '.params'
    file = open(f'results/{refGenome}/{analysis}_analysis{spec}/{prefix}.params', 'w')
    file_masked = open(f'results/{refGenome}/{analysis}_analysis{spec}/{prefix}_masked.params', 'w')
    structure = '[2' + ',2'*(len(pops.split('/'))-1) + ']'
    data = f'results/{refGenome}/{analysis}_analysis{spec}/{prefix}'

    # Rewrite this mess
    file_masked.write('Input data: ' + data +
                        '_masked.sfs\n\nEngine: ' + engine + '\n\nModel plot engine: demes' + '\n\nMutation rate: ' + 
                        repr(mutrate).replace('\'', '') + '\n\nSequence length: ' + seq_len + '\n\nTime for generation: ' + 
                        repr(gentime).replace('\'', '') + '\n\nInitial structure : ' + structure + '\n\nPts: ' + pts + 
                        '\n\nNumber of repeats: ' + str(repeats) + '\n\nNumber of processes: ' + processes + '\n' + 
                        '\nUnits of time in drawing: years' + '\n\nPrint models\' code every N iteration: 100')
    file.write('Input data: ' + data +
                '.sfs\n\nEngine: ' + engine + '\n\nModel plot engine: demes' + '\n\nMutation rate: ' + 
                repr(mutrate).replace('\'', '') + '\n\nSequence length: ' + seq_len + '\n\nTime for generation: ' + 
                repr(gentime).replace('\'', '') + '\n\nInitial structure : '+ structure + '\n\nPts: ' + pts + '\n\nNumber of repeats: ' + 
                str(repeats) + '\n\nNumber of processes: ' + processes + '\n' + '\nUnits of time in drawing: years' + '\n\nPrint models\' code every N iteration: 100')