Multiscale model for comorbid patients
=============================================================

The following mathematical models will be integrated:

- a PK/PD model of ACE inhibition of the renin-angiotensin system (RAS) [1][2][3][4][5]
- a subject-specific model of the circulatory system

Related Publications
---------------------

[1] Pilvankar, Minu R., Michele A. Higgins, and Ashlee N. Ford Versypt. "Mathematical Model for Glucose Dependence of the Local Renin–Angiotensin System in Podocytes." Bulletin of Mathematical Biology, 80, no. 4 (2018): 880-905.

[2] Pilvankar, M.R.; Higgins, M.A.; Ford Versypt, A.N. glucoseRASpodocytes. 2017. Available online: http:
//github.com/ashleefv/glucoseRASpodocytes (accessed on 31 March 2020).

[3] Pilvankar, Minu R., Hui L. Yong, and Ashlee N. Ford Versypt. "A Glucose-Dependent Pharmacokinetic/Pharmacodynamic Model of ACE Inhibition in Kidney Cells." Processes, 7(3), 131, 2019 https://doi.org/10.3390/pr7030131

[4] Ford Versypt, A.N.; Harrell, G.K.; McPeak, A.N. A pharmacokinetic/pharmacodynamic model of ACE
inhibition of the renin-angiotensin system for normal and impaired renal function. Comp. Chem. Eng.
2017, 104, 311–322.

[5] Ford Versypt, A.N. ACEInhibPKPD. 2017. Available online: http://github.com/ashleefv/ACEInhibPKPD
(accessed on 31 March 2020).

[6] Neal, Maxwell Lewis, and James B. Bassingthwaighte. "Subject-specific model estimation of cardiac output and blood volume during hemorrhage." Cardiovascular engineering 7.3 (2007): 97-120.

Quick start
-----------

You can install the software along with all its dependencies from
`GitHub <https://github.com/pietrobarbiero/COVID19-DKD-PKPD>`__:

.. code:: bash

    $ git clone https://github.com/pietrobarbiero/COVID19-DKD-PKPD.git
    $ cd ./COVID19-DKD-PKPD
    $ pip install -r requirements.txt .

After having installed all the requirements you can run
the example scripts:

- ``./examples/dkd.py`` for running the PK/PD model. The results will be saved under the directory``./examples/data/``. Once the results have been saved, you can make some plots by running the script ``./examples/make_plots.py``.
- ``./examples/cardio.py`` for running the cardiovascular model. Results will be inside ``./examples/cardio_y.csv`` and ``./examples/volumes.png``.

Contributing
-------------

The project (``./msmodel`` directory) has the following structure:

- the ``pk`` directory is used to specify the equations of the Pharmacokinetic model
- the ``pd`` directory is used to specify the equations of the Pharmacodynamic model
- the ``circulation`` directory is used to specify the equations of the circulatory system
- the ``ode`` directory is used to specify the ODE systems

The ``_config.py`` file is used to parse command line arguments.

The script ``_msmodel.py`` contains
the main interface function to run the ODE systems.


Source
------

The source code and minimal working examples can be found on
`GitHub <https://github.com/pietrobarbiero/COVID19-DKD-PKPD>`__.


Licence
-------

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.
