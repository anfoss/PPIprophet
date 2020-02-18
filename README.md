# APprophet

Software toolkit for analysis of affinity purified fractionated samples.
APprophet is a port of https://github.com/fossatiA/PCprophet


## Getting Started

These instructions will get you a copy of the project up and running on your local machine and how to test the compatibility with your current Python packages
### Prerequisites

* [Python >=3.4.x](https://www.python.org)
* [Sklearn 0.20.3](https://pypi.org/project/sklearn/)
* [NetworkX v2.4](https://networkx.github.io)
* [Pandas >0.23](https://pandas.pydata.org)
* [Scipy >1.1.0](https://www.scipy.org)
* [Igraph](https://igraph.org/python/)
* [Tensorflow2](https://www.tensorflow.org/install/)

### Installing

We recommend using [anaconda](https://www.anaconda.com) as it contains the majority of the required packages for APprophet.
Igraph needs to be installed separately [here](https://igraph.org/python/)
Tensorflow also requires separates installation [here](https://www.tensorflow.org)

#### Command line version

Ensure that you have installed the GitHub tool and 'git-lfs' command specifically for large file transfer. Please see [here](https://gist.github.com/derhuerst/1b15ff4652a867391f03) for installing GitHub and [here](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage) for installing 'git-lfs' command.

```
git clone https://github.com/fossatiA/APprophet APprophet
```

This will get you a working copy of APprophet into a folder called APprophet

> **note** for the command line version only Python3 and related package dependencies are necessary

## Usage

For usage of PCprophet refers to the [APprophet_instructions.md](https://github.com/fossatiA/PCprophet/blob/master/PCprophet_instructions.md)


## Contributing

Please read [CONTRIBUTE.md](https://github.com/fossatiA/PCprophet/blob/master/CONTRIBUTE.md) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Andrea Fossati**  - [fossatiA](https://github.com/fossatiA) fossati@imsb.biol.ethz.ch

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
