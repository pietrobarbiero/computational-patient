Computational patient
=============================================================

This repository contains the python code required to reproduce the
experiments presented in
"Barbiero, P. and Liò, P. (2020). The Computational Patient has Diabetes and a COVID".

Description
------------

Medicine is moving from reacting to a disease to prepare personalised and
precision paths to well being. The complex and multi level pathophysiological
patterns of most diseases require a systemic medicine approach and are challenging
current medical therapies.

Here we present a Digital patient model that integrates, refine and extend
recent specific mechanistic or phenomenological models ofcardiovascular [1],
RAS [2] and diabetic [3] processes. Our aim is twofold: analyse the modularity
and composability of the models-building blocks of the Digital patient and to
study the dynamical properties of well-being and disease states in a broader
functional context. We present results from a number of experiments among
which we characterise the dynamical impact of covid-19 and T2D diabetes on
cardiovascular and inflammaging conditions. We tested these experiments under
exercise and meals and drug regimen.

Common clinical parameters such as diastolic and systolic blood pressure,
heart patterns, blood cell counts are usually evaluated as averages.
Little importance is given to higher moments such as variances during the
day or during a longer interval of time. The lack of continuous measures
for most of the quantities has generated a medical practice that disregard
of unobserved or partially observed data. Our composable model reveals
interesting patterns, particularly fluctuations in blood pressure, particularly
when the diabetic model is coupled with the RAS and the cardiovascular models
under COVID acute infections.

References
-------------

[1] Neal, M. L., & Bassingthwaighte, J. B. (2007). Subject-specific model estimation of cardiac output and blood volume during hemorrhage. Cardiovascular engineering, 7(3), 97-120.

[2] Pilvankar, M. R., Yong, H. L., & Ford Versypt, A. N. (2019). A Glucose-Dependent Pharmacokinetic/Pharmacodynamic Model of ACE Inhibition in Kidney Cells. Processes, 7(3), 131.

[3] Topp, B., Promislow, K., Devries, G., Miura, R. M., & Finegood, D. T. (2000). A model of b-cell mass, insulin, and glucose kinetics: pathways to diabetes. Journal of theoretical biology, 206(4), 605.

Quick start
-----------

You can install the software along with all its dependencies from
`GitHub <https://github.com/pietrobarbiero/computational-patient>`__:

.. code:: bash

    $ git clone https://github.com/pietrobarbiero/computational-patient.git
    $ cd ./computational-patient
    $ pip install -r requirements.txt .

After having installed all the requirements you can run
the example script ``./examples/aging.py`` for running experiments.
The results will be saved under the directory ``./examples/data/``.

Once the results have been saved, you can make some plots
by running the script ``./examples/make_plots.py``.

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
`GitHub <https://github.com/pietrobarbiero/computational-patient>`__.


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
