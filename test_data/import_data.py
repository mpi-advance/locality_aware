import os
import ssgetpy as ss
import tarfile
import shutil
from convert_petsc import convert
import random

# Example parameters
# min_rowscols = 0
# max_rowscols = 300_000_000
# min_density = 0.0
# max_density = 0.5
# file_dir = './test_dir'

def _import_data():
    min_rowscols = int(input('Min dimensions:'))
    max_rowscols = int(input('Max Dimensions:'))
    min_density = float(input('Min density:'))
    max_density = float(input('Max density:'))
    file_dir = input('Output location:')
    search_limit = int(input('Search Limit:'))
    random_yesno = input('Random Results(y/n):')

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    results = ss.search(
            rowbounds = (min_rowscols, max_rowscols), colbounds = (min_rowscols, max_rowscols), limit=1500)

    for i in range(0, search_limit):
        # download
        if random_yesno.lower() == 'y':
            result = results.pop(random.randint(0,len(results)))
        else:
            result = results.pop(i)
        density = result.nnz / (result.rows * result.cols)
        if (density > min_density and density < max_density):
            result.download(destpath=f'{file_dir}')
            # extract
            with tarfile.open(f'{file_dir}/{result.name}.tar.gz', 'r:gz') as tar:
                tar.extractall(file_dir)

            shutil.move(f'{file_dir}/{result.name}/{result.name}.mtx', f'{file_dir}/{result.name}.mtx')
            shutil.rmtree(f'{file_dir}/{result.name}/')
            os.remove(f'{file_dir}/{result.name}.tar.gz')

            # convert
            convert(f'{file_dir}/{result.name}.mtx', f'{file_dir}/{result.name}.pm')

if __name__ == "__main__":
    _import_data()

