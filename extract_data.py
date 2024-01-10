#! /usr/bin/env python3

import os
import sys

def main():
    machine = sys.argv[1]
    experiment = os.getcwd().split("/")[-1].split(".")[0]
    with os.scandir(".") as it:
        for entry in it:
            if entry.is_file():
                with open(entry, 'r') as file:
                    print_string =""
                    data = file.read()
                    nodes, data = data.split("Modules")
                    nodes = nodes.split(":")[1].strip()
                    MPI, data = data.split("\n",1)
                    MPI = MPI.lstrip(': ')
                    for matrix in data.split("Starting ")[1:]:
                        matrix_name, matrix = matrix.split("\n",1)
                        matrix_name.strip()
                        for line in matrix.split("\n"):
                            if "form: " in line:
                                form_time = line.rstrip().split(": ")[1]
                            elif "comm: " in line:
                                p2p_time = line.rstrip().split(": ")[1]
                            elif "Standard" in line and "create: " in line:
                                mpi_time = line.rstrip().split(": ")[1]
                            elif "advance" in line and "create: " in line:
                                mpix_time = line.rstrip().split(": ")[1]

                        print(machine,nodes,matrix_name,MPI,experiment,p2p_time,mpi_time,mpix_time,sep=",")
                    print(print_string, end="")


if __name__ == "__main__":
	main()
