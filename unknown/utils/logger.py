class Logger():
  def __init__(self, job_file):
    self.job_file = job_file

  def log(self, *args):
    print(*args)
    with open(self.job_file, 'a+') as f:
      f.write( " ".join(["{}"]*len(args)).format(*args) )
      f.write("\n")
