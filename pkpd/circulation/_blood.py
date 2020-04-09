def _total_blood_volume(Vheart, VPulArt, Vpc, Vpv,
                        VSysArt, Vsc, VSysVen):
    """
    Equation A.100

    :param Vheart:
    :param VPulArt:
    :param Vpc:
    :param Vpv:
    :param VSysArt:
    :param Vsc:
    :param VSysVen:
    :return:
    """
    return Vheart + VPulArt + Vpc + Vpv + VSysArt + Vsc + VSysVen


def _heart_volume(Vra, Vrv, Vla, Vlv, Vcorcirc):
    """
    Equation A.101

    :param Vra:
    :param Vrv:
    :param Vla:
    :param Vlv:
    :param Vcorcirc:
    :return:
    """
    return Vra+Vrv+Vla+Vlv+Vcorcirc


def _coronary_volume(Vcorepi, Vcorintra, Vcorcap, Vcorvn):
    """
    Equation A.102

    :param Vcorepi:
    :param Vcorintra:
    :param Vcorcap:
    :param Vcorvn:
    :return:
    """
    return Vcorepi+Vcorintra+Vcorcap+Vcorvn


def _systemic_arterial_volume(Vaop, Vaod, Vsap, Vsa):
    """
    Equation A.103

    :param Vaop:
    :param Vaod:
    :param Vsap:
    :param Vsa:
    :return:
    """
    return Vaop + Vaod + Vsap + Vsa


def _systemic_venous_volume(Vsv, Vvc):
    """
    Equation A.104

    :param Vsv:
    :param Vvc:
    :return:
    """
    return Vsv + Vvc


def _pulmonary_arterial_volume(Vpap, Vpad, Vpa):
    """
    Equation A.105

    :param Vpap:
    :param Vpad:
    :param Vpa:
    :return:
    """
    return Vpap + Vpad + Vpa
