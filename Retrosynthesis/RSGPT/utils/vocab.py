mapping = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, 'C': 4, 'N': 5, 'O': 6, 'S': 7, 'H': 8, 'CL': 9, 'F': 10,
           'Br': 11, 'I': 12, 'Si': 13, 'P': 14, 'B': 15, 'Na': 16, 'K': 17, 'Al': 18, 'Ca': 19, 'Sn': 20, 'As': 21,
           'Hg': 22, 'Fe': 23, 'Zn': 24, 'Cr': 25, 'Se': 26, 'Gd': 27, 'Au': 28, 'Li': 29, '[MASK]': 30}

id2symbol = {v: k for k, v in mapping.items()}