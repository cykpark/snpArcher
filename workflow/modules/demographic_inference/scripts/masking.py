import dadi
import argparse

if __name__ == "__main__":
    try:
        from snakemake.script import snakemake
        sfs = snakemake.input['sfs']
        prefix = snakemake.params['prefix']
        pops = snakemake.params['pops']
        refGenome = snakemake.params['refGenome']
        analysis = snakemake.params['analysis']
        spec = snakemake.params['spec']

    except ImportError:
        parse = argparse.ArgumentParser()
        parse.add_argument('--popfile', '-p',type=str,action='store',help='popfile input file')
        parse.add_argument('--vcf', '-v',type=str,action='store',help='vcf input file')
        parse.add_argument('--outfile', '-o',type=str,action='store',help='output file')
        parse.add_argument('--populations', '-ps',type=str,action='store',help='population names')
        parse.add_argument('--refGenome', '-rg',type=str,action='store',help='reference genome')
        args = parse.parse_args()
        vcf = args.vcf
        popfile = args.popfile
        outfile = args.outfile
        pops = args.populations
        refGenome = args.refGenome
    """
    Mask FS
    """
    data_fs = dadi.Spectrum.from_file(sfs)
    if len(pops.split('/'))==3:
        data_fs.mask[0,0,:] = True
        data_fs.mask[:,0,0] = True
        data_fs.mask[0,:,0] = True
    elif len(pops.split('/'))==2:
        data_fs.mask[0,1] = True
        data_fs.mask[1,0] = True
    else:
        print('Could not mask due to population size incompatibility')

    data_fs.to_file(f'results/{refGenome}/{analysis}_analysis{spec}/{prefix}_masked.sfs')