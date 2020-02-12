import PCprophet.exceptions as PCpexc
import pandas as pd


class InputTester(object):
    """
    docstring for InputTester
    validate all inputs before anything
    infile is a panda dataframe
    """

    def __init__(self, path, filetype, infile=None):
        super(InputTester, self).__init__()
        self.infile = infile
        self.filetype = filetype
        self.path = path

    def read_infile(self):
        self.infile = pd.read_csv(self.path, sep="\t", index_col=False)

    def test_missing_col(self, col):
        """
        check columns in self
        """
        #  print(set(list(self.infile)))
        if not all([x in self.infile.columns for x in col]):
            raise PCpexc.MissingColumnError(self.path)

    def test_empty(self, col):
        for x in col:
            if self.infile[x].isnull().values.any():
                print(x)
                raise PCpexc.EmptyColumnError(self.path)

    def test_uniqueid(self, totest):
        if self.infile.duplicated(totest).any():
            print("The following rows in %s are duplicated".format(self.path))
            print(self.infile[self.infile.duplicated(totest)])
            raise PCpexc.DuplicateIdentifierError(self.path)

    def test_all(self, *args):
        """
        performs all test
        """
        self.test_missing_col(args[0])
        self.test_uniqueid(args[1])

    def test_na(self):
        if self.infile.isnull().values.any():
            raise PCpexc.NaInMatrixError(self.path)

    def test_file(self):
        self.read_infile()
        if self.filetype == "ids":
            col = ["Sample", "cond", "group", "short_id", "repl", "fr"]
            unique = ["repl", "short_id"]
            self.test_all(col, unique)
            self.test_empty(col)
        elif self.filetype == "db":
            try:
                col = ["ComplexID", "ComplexName", "subunits(Gene name)"]
                unique = ["ComplexName", "ComplexID"]
                self.test_all(col, unique)
                [self.test_empty(x for x in col)]
            except PCpexc.MissingColumnError as e:
                self.test_missing_col(["protA", "protB"])
        elif self.filetype == "in":
            col = ["GN", "ID"]
            unique = ["GN"]
            self.test_all(col, unique)
            self.test_na()
