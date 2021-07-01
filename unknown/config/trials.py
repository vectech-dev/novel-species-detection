""" provides which data is to be used in this trial (T1 was used in the paper)"""

trials = {
  'T1': {
    't1': {
      'algs': ['xception'],
      'num_classes': [16, 21],
      'layer_size': ['feats'],
      'combined': False
    },
    't2': ['rfor', 'svm', 'wnn'],
    't3': {
      'method': 'garb',
      'input': {
          't1': [],
          't2': ['all']
      }
    }
  }
}