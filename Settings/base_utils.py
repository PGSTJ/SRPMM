from . import csv



def _dct_read_csv(datafilepath:str) -> tuple[list[str], list[dict[str,str]]]:
    """ Return list of headers and LoD of row data """
    with open(datafilepath, 'r') as fn:
        header:list[str] = fn.readline().strip().split(',')
        rdr = csv.DictReader(fn, header)
        return header, [i for i in rdr]