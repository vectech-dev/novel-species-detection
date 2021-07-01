import json
import sys, os

known_taxa_file = os.getcwd()+ '/utils/known_taxa.json'
general_corrections_file = os.getcwd()+ '/utils/general_correction.json'

def get_corrections_data():
    with open(known_taxa_file, "r") as read_file:
        known_taxa = json.load(read_file)
    with open(general_corrections_file, "r") as read_file:
        general_corrections = json.load(read_file)
    return known_taxa, general_corrections

KNOWN_TAXA, GENERAL_CORRECTIONS = get_corrections_data()


class Specimen:
    def __init__(self, taxon):
        self.taxon = taxon.lower()
        self.wholecheck()
        self.space_fix()
        self.misspelled_genus = None
        self.genus = ''
        self.species = ''
        self.flags = []
        self.warnings = ''
        # Valid flags: GB = General Subspecies correction, GS = General Species correction,
        # GG = General genus correction, CG = Corrected Genus
        self.taxon_len = len(self.taxon.split(' '))
        # Extract genus and see if you need to fix any misspellings
        self.genus = self.taxon.split(' ')[0]
        if not self.genus in KNOWN_TAXA['__genus__']:
            self.genus_fix()

        # skip to the end if the General Genus flag was thrown
        if 'GG' not in self.flags:
            # Extract species and see if you need to fix any misspellings
            if self.taxon_len >= 2:
                if 'CG' in self.flags:
                    self.species = self.taxon.replace(self.misspelled_genus+ ' ','').replace(' ','_')
                else:
                    self.species = self.taxon.replace(self.genus+ ' ','').replace(' ','_')
                if self.species not in KNOWN_TAXA['__genus__'][self.genus]['__species__']:
                    self.species_fix()

            else:
                self.species = 'spp'

        self.taxon = self.genus + ' ' + self.species
        self.space_fix()
        if not self.warnings == '':
            print('WARNING: '+ self.warnings)

    def space_fix(self):
        IDed = False
        for a in range(len(self.taxon)):
            if not self.taxon[a] == ' ':
                IDed = True
                break
            else:
                IDed = False
        if IDed:
            while ' ' == self.taxon[0]:
                self.taxon = self.taxon[1:]
            while ' ' == self.taxon[-1]:
                self.taxon = self.taxon[:-1]
            while '  ' in self.taxon:
                self.taxon = self.taxon.replace('  ',' ')
        else:
            self.taxon = 'no_id'

    def genus_fix(self):
        found = False
        self.misspelled_genus = self.genus
        for genkey in KNOWN_TAXA['__genus__']:
            for misspell in KNOWN_TAXA['__genus__'][genkey]['__misspelling__']:
                # If you find the genus in the misspelling list, then correct it to the current genkey genus.
                if self.genus == misspell:
                    self.genus = genkey
                    found = True
                    self.flags.append('CG')
        if not found:
            if self.genus in GENERAL_CORRECTIONS['genus']:
                found = True
                self.flags.append('GG')
            for cor_key in GENERAL_CORRECTIONS['genus']:
                if self.genus in GENERAL_CORRECTIONS['genus'][cor_key] or self.genus == cor_key:
                    self.genus = cor_key
                    found = True
                    self.flags.append('GG')
                    break
            if not found:
                sys.exit('ERROR: Please analyze the following genus misspelling and add the correct info into the '
                         'KNOWN_TAXA variable in this script:\n' + self.genus)

    def species_fix(self):
        found = False
        for spkey in KNOWN_TAXA['__genus__'][self.genus]['__species__']:
            for misspell in KNOWN_TAXA['__genus__'][self.genus]['__species__'][spkey]['__misspelling__']:
                # If you find the species in the misspelling list, then correct it to the current spkey species.
                if self.species == misspell:
                    self.species = spkey
                    found = True
                    break
        if not found:
            for cor_key in GENERAL_CORRECTIONS['species']:
                if self.species in GENERAL_CORRECTIONS['species'][cor_key] or self.species == cor_key:
                    self.species = cor_key
                    found = True
                    self.flags.append('GS')
                    break

            if not found:
                sys.exit('ERROR: Please analyze the following species misspelling and add the correct info into the '
                     'KNOWN_TAXA variable in this script:\n--' + self.genus + '_' + self.species+'--')

    def subspecies_fix(self):
        found = False
        if type(KNOWN_TAXA['__genus__'][self.genus]['__species__'][self.species]['__subspecies__']) is dict:
            for subkey in KNOWN_TAXA['__genus__'][self.genus]['__species__'][self.species]['__subspecies__']:
                for misspell in KNOWN_TAXA['__genus__'][self.genus]['__species__'][self.species]['__subspecies__'][subkey]['__misspelling__']:
                    # If you find the subpecies in the misspelling list, then correct it to the current subkey subspecies.
                    if self.subspecies == misspell:
                        self.subspecies = subkey
                        found = True
                        break
        if not found:
            for cor_key in GENERAL_CORRECTIONS['subspecies']:
                if self.subspecies in GENERAL_CORRECTIONS['subspecies'][cor_key] or self.subspecies == cor_key:
                    self.subspecies = cor_key
                    found = True
                    self.flags.append('GB')
                    break
            if not found:
                sys.exit('ERROR: Please analyze the following subspecies misspelling and add the correct info into the '
                         'KNOWN_TAXA variable in this script:\n--' + self.genus + '_' + self.species + '_' +
                         self.subspecies + '--')

    def wholecheck(self):
        # Latin abbreviations modifications
        if 'nr. ' in self.taxon:
            self.taxon = self.taxon.replace('nr. ', 'nr-')
        if 'cf. ' in self.taxon:
            self.taxon = self.taxon.replace('cf. ', 'cf-')

        # General removals
        if '(morphology)' in self.taxon:
            self.taxon = self.taxon.replace('(morphology)', '')
        if 'sp.' in self.taxon:
            self.taxon = self.taxon.replace('sp. ', '')
        if 'nfl-2015' in self.taxon:
            self.taxon = self.taxon.replace('nfl-2015', '')



############################################
#   Correction Functions
############################################

def correct(taxon):
    # This is for formats giving and receiving a full taxon.
    specimen1 = Specimen(taxon)
    return specimen1.taxon


def correct2(genus,species_and_subspecies):
    # This is for formats where the species and subspecies come concatenated by a space, as is common to the
    # JHI Image database and the bold full data spreadsheets.

    taxon = genus + ' ' + species_and_subspecies
    specimen1 = Specimen(taxon)
    return specimen1.genus, specimen1.species, specimen1.subspecies

def correct3(genus,species, subspecies):
    # This is for formats where the species and subspecies and genus are all separate.

    taxon = genus + ' ' + species + ' ' + subspecies
    specimen1 = Specimen(taxon)
    return specimen1.genus, specimen1.species, specimen1.subspecies


def check(taxon):
    specimen1 = Specimen(taxon)
    if specimen1.taxon == taxon:
        return True
    else:
        return False

def test():
    moz1 = corrections.Specimen('deinocerites sp. CUBA-1')
    moz2 = corrections.Specimen('anopheles crucian complex(cr. c.)')
    moz3 = corrections.Specimen('Aedes ')
    moz4 = corrections.Specimen('Culex unknown')
    moz5 = corrections.Specimen('anopheles crucians complex e')
    moz6 = corrections.Specimen('anopheles crucians complex E')
    moz7 = corrections.Specimen('anopheles crucians E')
    moz8 = corrections.Specimen('anopheles crucians')
    moz9 = corrections.Specimen(' ')
    moz10 = corrections.Specimen('')
    moz11 = corrections.Specimen('void')
    moz12 = corrections.Specimen('An. funestus s.l. (morphology)')
    moz13 = corrections.Specimen('An. funestus')
    moz14 = corrections.Specimen('An. rivulorum')

    print(moz1.taxon)
    print(moz2.taxon)
    print(moz3.taxon)
    print(moz4.taxon)
    print(moz5.taxon)
    print(moz6.taxon)
    print(moz7.taxon)
    print(moz8.taxon)
    print(moz9.taxon)
    print(moz10.taxon)
    print(moz11.taxon)
    print(moz12.taxon)
    print(moz13.taxon)
    print(moz14.taxon)
