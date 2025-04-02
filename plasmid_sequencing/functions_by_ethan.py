'''some functions ethan wrote for seq data'''

def read_list(file):
    '''
    input:
        file - path to file to read in, should be new line between each entry

    returns:
        list - a list of the entries in file

    this version strips bytemarker characters
    
    '''
    return open(file, 'r').read().strip('\ufeff').split('\n')

def save_list(out_file, listlike):
    '''
    saves a list of inputs with a newline between each list entry
    input:
        outfile - path to file to save,
        listlike - the list to save

    returns:
    
    '''
    return open(outfile, 'w').write(''.join([str(i) + '\n' for i in listlike]))

                                    

def write_fasta(outfile, fasta_dict):
    '''
    saves a dictionary as a fasta file

    input:
        outfile - a path to the output fasta
        fasta_dict - a dictionary where the key is an entry name
                    and the value is a genomic or protein sequence
                    corresponding to the key name in the fasta
    '''
    fasta_string = ''.join(['>' + str(name) + '\n' + str(seq) + '\n' for name, seq in fasta_dict.items()])
    return open(outfile, 'w').write(fasta_string)
    