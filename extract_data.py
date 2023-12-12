#! /usr/bin/env python3

import os

def main():
    matrix = None
    with os.scandir(".") as it:
        for entry in it:
            if entry.is_file():
                with open(entry, 'r') as file:
                    print_string =""
                    for line in file:
                        if "Nodes:" in line:
                            nodes = line.rstrip().split(": ")[1]
                        elif "Starting " in line and "echo" not in line:
                            if matrix is not None:
                                print_string += (" " + matrix + "\n")
                            matrix = line.rstrip().split("Starting ")[1]
                            print_string += (nodes + " ") 
                        elif "form: " in line or "comm: " in line or "create: " in line or "neighbor: " in line:
                            print_string += (" " + line.rstrip().split(": ")[1])
                    print(print_string, end="")


if __name__ == "__main__":
	main()
