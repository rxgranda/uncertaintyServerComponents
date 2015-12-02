# Uncertainty Dashboard Backend

Implementation of a component based prediction model. This implementation make use of *machine learning* techniques in order to manage the *uncertainty* in an academic enviroment for the decision process of course selection for a student.

## Component Diagram

<img src="https://raw.githubusercontent.com/rxgranda/uncertaintyServerComponents/master/doc/Diagram_componentes_v1.2.png">

## Installation

Be sure to installed [R](https://cran.r-project.org/bin/).
And for linux have installed **liblzma**, here the instructions for debian based systems:

```bash
apt-get update
apt-get install r-base r-base-dev liblzma5 liblzma-dev
```
Them clone the git repository and install the python dependences:
```bash
git clone https://github.com/rxgranda/uncertaintyServerComponents.git
cd uncertaintyServerComponents
pip install -r requirements.txt
```

Run the script for install the R required packages:
```bash
./r_requirements_install.py
```

Finally import the data from the remote database:
```bash
./query2csv.py
```
If there exits a problem a zip file is alocated in this [link](https://drive.google.com/a/cti.espol.edu.ec/file/d/0B9kjd1_TSf0VMHk0N2R1R21QbGM/view?userstoinvite=fsalvador23@gmail.com&ts=5658873e&actionButton=1), and needs to be uncompresed in the **data/** folder.

## API Reference

Documentation can be find in the **doc/** folder.

## Tests

For an example code of the prediction model instantiation can be found in the **test_script/** folder, simply run:
```bash
./test_scripts/classifier_test.py
```

## License

A short snippet describing the license (MIT)
