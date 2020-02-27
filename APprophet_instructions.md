## APprophet (command line version)

In the APprophet package, all parameters can be configured either via the ‘ProphetConfig.conf’ file or via by running APprophet using the command. When running the APprophet, the parameters indicated in the command will be written into the ‘ProphetConfig.conf’ file. Generally, four types of features are needed:


## Input file

Input file need to be a wide format matrix with the format


|GN   |Fraction1|
|:----|:--------|
|A    |20400    |
|D    |1230120  |
|C    |1230120   |

GN can be **any** identifier given that is unique for that particular run

Remaining columns needs to be ordered according to the fractionation scheme used. *There is no strict requirement for column names apart from GN, but they need to be ordered*
All quantitation schemes commonly used in proteomics such as MS1 or MS2 ion-extracted chromatogram (XIC), spectral counts (SPCs) and TMT or SILAC ratios are supported.

Examples of correct formatting are provided in the test_fract.txt data


### Contact
Please refer to [README.md](https://github.com/fossatiA/APprophet/blob/master/README.md) for how to contact us
