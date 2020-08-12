#!/bin/bash
#SBATCH -J sm
#SBATCH -o out
#SBATCH -e err
#SBATCH -p test
#SBATCH -n 1
#SBATCH -t 400
#SBATCH --mem=4000


#snakemake --snakefile Snakefile_bam2vcf_gatk --profile ./profiles/slurm
snakemake --snakefile Snakefile_bam2vcf_fb --profile ./profiles/slurm 

