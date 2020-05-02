import os

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt

from ..circulation._baroreceptor import _f, _b_vaso, _firing_frequency, _n_change
from ..circulation._blood import _total_blood_volume, _heart_volume, _coronary_volume, _systemic_arterial_volume, \
    _systemic_venous_volume, _pulmonary_arterial_volume
from ..circulation._coronary import _p_eq, _pcor, _pcorc, _fcor, _vcor
from ..circulation._heart import _trigger_A, _trigger_B, t_rel, _yi, _ei, _v0, _pa, _fi, _volume_change, _pv, _emaxv
from ..circulation._pulmonary import _ppap, _ppad, _pp, _fp, _vp, _fpa
from ..circulation._systemic import _cardiac_output, _stroke_volume, _arterial_blood_pressure, _kv, _map, \
    _aortic_afterload, _psap, \
    _psa_a, _psa_p, _psa, _psc, _psv, _pvc, _forward_flow, _abp_change, _aortic_flow_change, _vaop_change, _v_change, \
    _pulmonary_valve_flow_change, _pressure_aop_change, _rsa, _rvc, _mapmeas_change, _comea_change

# unit conversion
to_L_min = 10 ** (-3) * 60
to_ml_sec = 10 ** 3 / 60
to_ml = 10 ** 3
to_min = 60


def ODE(t, y,

        x, tHB, HP, tmeas, ABPmeas, PAFmeas, Nbr_list, Nbr_list_idx,
        # HRa, HRv, Tsv, Tsa, Vvarlvs0, Vvarrvs0, n, m, tRwave, tPwave, af_con2,
        change_list,
        t_list,
        Pra_list, Prv_list, Pla_list, Plv_list,
        Vra_list, Vrv_list, Vla_list, Vlv_list,

        # heart
        Ts1v, Ts1a, Ts2, offv,
        Vlvd0, Vlvs0, Vrvd0, Vrvs0, Vlad0, Vlas0, Vrad0, Vras0,
        Rra, Rla, Rlv, Rrv,
        PRint, KElv, KErv,
        Emaxlv1, Eminlv, Emaxrv1, Eminrv,
        EDVLV, EDVRV,
        Emaxra, Eminra, Emaxla, Eminla,

        # systemic
        KCOMAP, Raop, Rtaop, Rcrb, Raod, Rtaod, Rsap, Rsc, Rsv,
        Caop, Caod, Csap, Csc,
        Vaop0, Vaod0, Vsap0, Vsc0,
        Laop, Laod,
        Kc, Do, Vsa0, Vsa_max, Kp1, Kp2, Kr, Rsa0, tau_p, Ksv, Kv1, Vmax_sv, D2, K1, K2, KR, R0, Vvc0, Vmax_vc, Vmin_vc,
        tauCO, Kxp, Kxv, Kxv1, Kxp1, tauMAP, tauABP,

        # pulmonary
        Rtpap, Rtpad, Rpap, Rpad, Rps, Rpa, Rpc, Rpv,
        Cpap, Cpad, Cpa, Cpc, Cpv,
        Vpap0, Vpad0, Vpa0, Vpc0, Vpv0,
        Lpap, Lpad,

        # coronary
        Rcorepi, Rcorintra, Rcorcap, Rcorvn,
        Ccorepi, Ccorintra, Ccorcap, Ccorvn,
        Vcorepi0, Vcorintra0, Vcorcap0, Vcorvn0,

        # baroreceptor
        a, a1, a2, K,
        K_con, T_con, l_con, a_con, b_con, tau_con, No_con,
        K_vaso, T_vaso, l_vaso, a_vaso, tau_vaso, No_vaso,
        amin, bmin, Ka, Kb
        ):
    Vra, Vrv, Vla, Vlv, \
    MAPmeas, Faop, Faod, Frv_sm, Vaop, Vaod, Vsa, Vsap, Vsc, Vsv, Vvc, Paop, AOFmod, ABPfol, COmea, \
    Vpap, Vpad, Vpa, Vpc, Vpv, Fpap, Fpad, \
    Vcorepi, Vcorintra, Vcorcap, Vcorvn, \
    Nbr, Nbr_t, N_con, N_vaso = y

    # # variables that may need to be updated at each step
    # cl = change_list[-1]
    # HRa, HRv, Tsv, Tsa, Vvarlvs0, Vvarrvs0, n, m, tRwave, tPwave, af_con2 = cl
    # n = int(n)
    # m = int(m)
    #
    # # set basic variables
    # f_con = a_con + b_con / (np.exp(tau_con * (N_con - No_con)) + 1.0)
    # af_con = amin + (Ka * f_con)
    # ABPshift = np.interp(t - offv, tmeas, ABPmeas)  # TODO: double check

    # triggers
    if t + dt >= tshift + 1/HR:
        Tsv = TsvK * np.sqrt(1 / HR / 60)
        Tsa = TsaK * np.sqrt(1 / HR / 60)
        HR = HRcont
        tshift = t
        afs_con2 = afs_con

    if np.interp(t - dt, t_list, ylv) == 0 and ylv > 0:
        EDPLV = np.interp(t - dt, t_list, Plvc)
        EDPRV = np.interp(t - dt, t_list, Prvc)
        EDVLV = np.interp(t - dt, t_list, Vlv)
        EDVRV = np.interp(t - dt, t_list, Vrv)

    trela = t - tshift
    trelv = t - tshift - PRint

    if 0.0<=trela and trela<Tsa:
        yla = (1.0 - np.cos(np.pi*trela/Tsa))/2.0
    elif Tsa<=trela and trela<1.5*Tsa:
        yla = (1.0 + np.cos(2.0*np.pi*(trela-Tsa)/Tsa))/2.0
    else:
        yla = 0
    yra = yla

    if 0.0 <= trelv and trelv < Tsv:
        ylv = (1.0 - np.cos(np.pi*trelv/Tsv))/2.0
    elif Tsv<=trelv and trelv<1.5*Tsv:
        ylv = (1.0 + np.cos(2.0*np.pi*(trelv-Tsv)/Tsv))/2.0
    else:
        ylv = 0
    yrv = ylv

    if t>tshift and t<tshift + 0.02:
        P_QRSwave = 5
    elif t>(tshift+PRint) and t<(tshift+PRint+0.02):
        P_QRSwave = 10
    else:
        P_QRSwave = 0

    # ----------------- heart ----------------- #

    Era= (Emaxra-Eminra)*yra+Eminra
    Vrar= (1-yra)*(Vrard-Vrars) + Vrars
    Erv = ((Emaxrv - Eminrv) * yrv) + Eminrv
    Vrvr = (1 - yrv) * (Vrvrd - Vrvrs) + Vrvrs
    Ela = (Emaxla - Eminla) * yla + Eminla
    Vlar = (1 - yla) * (Vlard - Vlars) + Vlars
    Elv = ((Emaxlv - Eminlv) * ylv) + Eminlv
    Vlvr = (1 - ylv) * (Vlvrd - Vlvrs) + Vlvrs
    Pla = (Vla - Vlar) * Ela - Px2 * (1 / (exp((Vla) / Vx8) - 1))
    Plac = Pla + Ppcdc
    if Plac > Plvc:
        Fla = (Plac - Plvc) / Rla
    else:
        Fla = 0
    Plv = afs_con2 * (Vlv - Vlvr) * Elv - Px2 * (1 / (exp((Vlv) / Vx8) - 1))
    Plvc = Plv + Ppcdc
    if Plvc>Paopc:
        Flv = (Plvc-Paopc)/Rav
    else:
        Flv = 0
    Pra = (Vra - Vrar) * Era - Px2 * (1 / (exp((Vra) / Vx8) - 1))
    Prac = Pra + Ppcdc
    if Prac>Prvc:
        Fra = (Prac-Prvc)/Rra
    else:
        Fra = 0
    Prv = afs_con2 * (Vrv - Vrvr) * Erv - Px2 * (1 / (exp(Vrv / Vx8) - 1))
    Prvc = Prv + Ppcdc
    if Prvc>Ppapc:
        Frv = (Prvc - Ppapc) / Rpuv
    else:
        Frv = 0


    # ----------------- systemic ----------------- #

    Rsa = Rsao + (Kr * exp(4 * F_vaso)) + (Kr * (Vsa_max / Vsa) ^ 2)
    Rvc = (KR * (Vmax_vc / Vvc) ^ 2) + Ro
    Paop = Paopc - PplcFOL
    MAP_dt = (Psa - MAP) / COtau
    Paodc = (Rtaod * Rcrb * Faop - Rtaod * Rcrb * Faod + (Vaod * Rcrb / Caod) - Rcrb * Px2 / (exp(Vaod / Vx8) - 1) + Pvcc * Rtaod) / (Rcrb + Rtaod)
    Paod = Paodc - Pbs
    PaodFOL_dt = (Paod - PaodFOL) / (0.0005sec)
    Psa_a = Kc * log(((Vsa - Vsa_o) / Do) + 1)
    Psa_p = Kp1 * exp(tau_p * (Vsa - Vsa_o)) + Kp2 * (Vsa - Vsa_o) ^ 2
    Psa = F_vaso * Psa_a + (1 - F_vaso) * Psa_p
    Psap = Vsap / Csap - Px2 * (1 / (exp(Vsap / Vx8) - 1))
    Psc = Vsc / Csc - Px2 * (1 / (exp(Vsc / Vx8) - 1))
    Psv = -Kv * log((Vmax_sv / Vsv) - 0.99)
    if Vvc>Vo:
        Pvc = D2+K2*exp(Vo/Vmin_vc)+K1*(Vvc-Vo) - Px2/(exp(Vvc/Vx8)-1)
    else:
        Pvc = D2+K2*exp(Vvc/Vmin_vc) - Px2/(exp(Vvc/Vx8)-1)
    Pvcc = Pvc + Pplc
    Fcrb = (Paodc - Pvcc) / Rcrb
    Fsap = (Psap - Psa) / Rsap
    Fsa = (Psa - Psc) / Rsa
    Fsc = (Psc - Psv) / Rsc
    Fsv = (Psv - Pvcc) / Rsv
    Fvc = (Pvcc - Prac) / Rvc


    # ----------------- pulmonary ----------------- #
    Ppapc1 = (Rpuv * (Vpap / Ctpap) - Rpuv * (Px2 * (1 / (exp(Vpap / Vx8) - 1))) + Prvc * Rtpap - Rpuv * Rtpap * Fpap + Pplc * Rpuv) / (Rtpap + Rpuv)
    Ppapc2 = (Vpap / Ctpap + Pplc - Rtpap * Fpap - Px2 * (1 / (exp(Vpap / Vx8) - 1)))
    if Prvc>Ppapc1:
        Ppapc = Ppapc1
    else:
        Ppapc = Ppapc2
    Ppap = Ppapc - Pplc
    Ppadc = Vpad_dt * Rtpad + Pplc + Vpad / Ctpad - Px2 * (1 / (exp(Vpad / Vx8) - 1))
    Ppad = Ppadc - Pplc
    Ppa = Vpa / Cpa - Px2 * (1 / (exp(Vpa / Vx8) - 1))
    Ppac = Ppa + Pplc
    Ppc = Vpc / Cpc - Px2 * (1 / (exp(Vpc / Vx8) - 1))
    Ppcc = Ppc + Pplc
    Ppv = Vpv / Cpv - Px2 * (1 / (exp(Vpv / Vx8) - 1))
    Ppvc = Ppv + Pplc
    Fps = (Ppac - Ppvc) / Rps
    Fpa = (Ppac - Ppcc) / Rpa
    Fpc = (Ppcc - Ppvc) / Rpc
    Fpv = (Ppvc - Plac) / Rpv

    # ----------------- baroreceptor ----------------- #
    Nbr_dt = Nbr_t
    (a2 * a * (Nbr_t_dt)) + ((a2 + a) * Nbr_t) + Nbr = (K * (Paod) + (a1 * K * PaodFOL_dt))
    t_min = 0
    if t + t_min > L_hrv:
        N_hrv_dt = (-N_hrv + (K_hrv*Nbr(t-L_hrv)))/T_hrv
    else:
        N_hrv_dt = 0
    F_hrv = a_hrv + (b_hrv / (exp(tau_hrv * (N_hrv - No_hrv)) + 1.0))
    b_hrv = 1 - a_hrv
    if t_min+t>L_hrs:
        N_hrs_dt = (-N_hrs + (K_hrs * Nbr(t - L_hrs))) / T_hrs
    else:
        N_hrs_dt = 0
    F_hrs = a_hrs + (b_hrs / (exp(tau_hrs * (N_hrs - No_hrs)) + 1.0))
    b_hrs = 1 - a_hrs
    if t_min+t>L_con:
        N_con_dt = (-N_con + (K_con * Nbr(t - L_con))) / T_con
    else:
        N_con_dt = 0
    F_con = a_con + (b_con / (exp(tau_con * (N_con - No_con)) + 1.0))
    if t_min+t>L_vaso:
        N_vaso_dt = (-N_vaso + (K_vaso * Nbr(t - L_vaso))) / T_vaso
    else:
        N_vaso_dt = 0
    b_vaso = 1 - a_vaso
    F_vaso = a_vaso + (b_vaso / (exp(tau_vaso * (N_vaso - No_vaso)) + 1.0))
    HRcont = (h1 + (h2 * F_hrs) - (h3 * F_hrs ^ 2) - (h4 * F_hrv) + (h5 * F_hrv ^ 2) - (h6 * F_hrv * F_hrs))
    afs_con = amin + (Ka * F_con)
    bfs_con = bmin + (Kb * F_con)


    # ----------------- airway mechanics ----------------- #
    if t>=tresp+(1/RespR):
        RespR = RespRcont
        tresp = t
    Rco = 0
    if Vc>Vcmax:
        Rc = Kc_air + Rco
    else:
        Rc = Kc_air * (Vcmax / Vc) ^ 2 + Rco
    Rs = As * exp(Ks * (VA - RV) / (Vstar - RV)) + Bs
    Ru = Au + Ku * abs(dV_dt)
    A = KA * gc + DA
    B = KB * gc + DB
    C = KC * gc + DC
    Pl = Al * exp(Kl * VA) + Bl
    Pcw = (A * sin(2 * PI * RespR * (t - tresp)) - B)
    if Vc / Vcmax < 0.5:
        Pc = Ac-Bc*(Vc/Vcmax - 0.7)^2
    else:
        Pc = Ac-Bc*(0.5 - 0.7)^2-Bcp*ln(Vcmax/Vc - 0.999)
    Ppl = Pcw
    Pve = Vve / Cve
    PplmmHg = Ppl
    Pmouth = Pbs
    Pplc = Ppl
    PplcFOL_dt = (Pplc - PplcFOL) / (0.001sec)
    if C/(2*PI)>(120min^-1):
        RespRcont = 120min^-1
    else:
        C / (2 * PI)
    Pcc = Pplc + Pc
    Ps = Pcc - PA
    Qdotco = (Pmouth - Pcc) / (Ru + Rc)
    Qdotup = Qdotco
    Qdotve = Pve / Rve
    Qdotsm = Ps / Rs
    Alvflux = (Pstp * Tbody / ((PA + (760 mmHg)) * Tstp)) * (O2flux + CO2flux)
    PA = Ppl + Pve + Pl
    PD = Qdotup * Ru
    PDc = Qdotco * Rc + Pcc
    Pup = Qdotup * (Ru + Rc)



    # ----------------- gas exchange ----------------- #
    SHbO2_ao = (PO2_ao / P50_O2) ^ nH / (((PO2_ao / P50_O2) ^ nH) + 1)
    SHbO2_pa = (PO2_pa / P50_O2) ^ nH / (((PO2_pa / P50_O2) ^ nH) + 1)
    dSHbO2dPO2_ao = nH * (PO2_ao ^ -1) * (PO2_ao / P50_O2) ^ nH / (((PO2_ao / P50_O2) ^ nH) + 1) ^ 2
    dSHbO2dPO2_pa = nH * (PO2_pa ^ -1) * (PO2_pa / P50_O2) ^ nH / (((PO2_pa / P50_O2) ^ nH) + 1) ^ 2
    SHbO2_aoISR = SHbO2_ao * 100
    CtO2_ao = alphaO2 * PO2_ao + CHb * SHbO2_ao * Hcrit
    CtO2_pa = alphaO2 * PO2_pa + CHb * SHbO2_pa * Hcrit
    CtO2_isf = alphaO2 * PO2_isf
    if Vpc>Vpcmax
        DL_O2 = ((0.397 mlSTPD/s/mmHg)+((0.0085 mlSTPD/s/mmHg^2)*PO2_ao)-((0.00013 mlSTPD/s/mmHg^3)*(PO2_ao^2)))+((5.1e-7 mlSTPD/s/mmHg^4)*(PO2_ao^3))
    else:
        DL_O2 = (sqrt(Vpc/Vpcmax))*((0.397 mlSTPD/s/mmHg)+((0.0085 mlSTPD/s/mmHg^2)*PO2_ao)-((0.00013 mlSTPD/s/mmHg^3)*(PO2_ao^2)))+((5.1e-7 mlSTPD/s/mmHg^4)*(PO2_ao^3))
    if PO2_isf<=0:
        V_O2 = 0
    else:
        V_O2 = (Vcytox * tweight * alphaO2 * PO2_isf) / (Kcytox + (alphaO2 * PO2_isf))
    O2flux = DL_O2 * (PA_O2 - PO2_ao) * (22400 ml / mole)
    SHbCO2_ao = (PCO2_ao / P50_CO2) / ((PCO2_ao / P50_CO2) + 1)
    SHbCO2_pa = (PCO2_pa / P50_CO2) / ((PCO2_pa / P50_CO2) + 1)
    dSHbCO2dPCO2_ao = (PCO2_ao ^ -1) * (PCO2_ao / P50_CO2) / ((PCO2_ao / P50_CO2) + 1) ^ 2
    dSHbCO2dPCO2_pa = (PCO2_pa ^ -1) * (PCO2_pa / P50_CO2) / ((PCO2_pa / P50_CO2) + 1) ^ 2
    CtCO2_ao = alphaCO2 * PCO2_ao + CHb * SHbCO2_ao * Hcrit
    CtCO2_pa = alphaCO2 * PCO2_pa + CHb * SHbCO2_pa * Hcrit
    CtCO2_isf = alphaCO2 * PCO2_isf
    if Vpc>Vpcmax:
        DL_CO2 = 16.67 mlSTPD / s / mmHg
    else:
        DL_CO2 = (sqrt(Vpc/Vpcmax))*(16.67 mlSTPD/s/mmHg)
    V_CO2 = RQ * V_O2
    CO2flux = DL_CO2 * (PA_CO2 - PCO2_ao) * (22400 ml / mole)
    Pao_O2 = (Pao - PH2O) * r_Pao_O2
    Pao_CO2 = (Pao - PH2O) * r_Pao_CO2
    Pao_N2 = (Pao - PH2O) - Pao_O2 - Pao_CO2
    Pmouth_CO2 = PD_CO2
    PD_N2 = (PDc + Pao) - PD_O2 - PD_CO2
    PC_N2 = (Pcc + Pao) - PC_O2 - PC_CO2
    PA_N2 = (PA + Pao) - PA_O2 - PA_CO2
    if Vpc>Vpcmax:
        DL_N2 = (0.25 mlSTPD/s/mmHg)
    else:
        DL_N2 = (sqrt(Vpc/Vpcmax))*(0.25 mlSTPD/s/mmHg)



    # ----------------- chemoreceptors ----------------- #
    fc = (Kche + ((1.4mmHg ^ (-1)) * PCO2_ao * exp(-PO2_ao / Pche))) / 100
    if PCO2_ao>=20:
        Kche = (0.35mmHg^(-1))*(PCO2_ao-20)
    else:
        Kche = 0


    # blood gas handling
    if (((Vcytox)/(Kcytox + CtO2_isf))>(51ml/min/g)):
        GO2isf = (51ml/min/g)
    else:
        GO2isf = (Vcytox) / (Kcytox + CtO2_isf)
    DVRO11 = 1 + dSHbO2dPO2_ao * O2cap * Hbconc / alphaO2
    DVRO21 = 1 + dSHbO2dPO2_pa * O2cap * Hbconc / alphaO2
    pH_ao = -log(H_ao / HpNorm)
    pH_pa = -log(H_pa / HpNorm)
    pH_isf = -log(H_isf / HpNorm)



    # ----------------- pericardium ----------------- #
    Ppcd = (K_pcd * exp((Vpcd - Vpcd_o) / phi_pcd)) - Px2 * (1 / (exp(Vpcd / Vx75) - 1))
    Ppcdc = Ppcd + Pplc
    Vpcd = Vrv + Vra + Vlv + Vla + perifl + Vmyo + Vcorcirc


    # ----------------- coronary circulation ----------------- #
    Pcorisfc = abs((Plvc - Ppcdc) / 2)
    Fcorao = (Pcoraoc - Pcoreac) / Rcorao
    Fcorea = (Pcoreac - Pcorlac) / Rcorea
    Fcorla = (Pcorlac - Pcorsac) / Rcorla
    Fcorsa = (Pcorsac - Pcorcapc) / Rcorsa
    Fcorcap = (Pcorcapc - Pcorsvc) / Rcorcap
    Fcorsv = (Pcorsvc - Pcorlvc) / Rcorsv
    Fcorlv = (Pcorlvc - Pcorevc) / Rcorlv
    Fcorev = (Pcorevc - Prac) / Rcorev
    Pcorao = Paop
    Pcorea = Vcorea / Ccorea - Px1 * (1 / (exp(Vcorea / Vx1) - 1))
    Pcorla = Vcorla / Ccorla - Px1 * (1 / (exp(Vcorla / Vx1) - 1))
    Pcorsa = Vcorsa / Ccorsa - Px1 * (1 / (exp(Vcorsa / Vx1) - 1))
    Pcorcap = Vcorcap / Ccorcap - Px1 * (1 / (exp(Vcorcap / Vx1) - 1))
    Pcorsv = Vcorsv / Ccorsv - Px1 * (1 / (exp(Vcorsv / Vx1) - 1))
    Pcorlv = Vcorlv / Ccorlv - Px1 * (1 / (exp(Vcorlv / Vx1) - 1))
    Pcorev = Vcorev / Ccorev - Px2 * (1 / (exp(Vcorev / Vx8) - 1))
    Pcoraoc = Paopc
    Pcoreac = Pcorea + Ppcdc
    Pcorlac = Pcorla + Pcorisfc
    Pcorsac = Pcorsa + Pcorisfc
    Pcorcapc = Pcorcap + Pcorisfc
    Pcorsvc = Pcorsv + Ppcdc
    Pcorlvc = Pcorlv + Ppcdc
    Pcorevc = Pcorev + Ppcdc
    Vcorcirc = Vcorao + Vcorea + Vcorla + Vcorsa + Vcorcap + Vcorsv + Vcorlv + Vcorev



    # ----------------- Differential Equations ----------------- #

    # heart
    Vla_dt = (Fpv - Fla)
    Vlv_dt = (Fla - Flv)
    Vra_dt = (Fvc - Fra + Fcorev)
    Vrv_dt = (Fra - Frv)
    COutput_dt = (Flv - COutput) / COtau
    SV = COutput / HR

    # systemic
    Paopc_dt = (Flv - Vaop_dt - Faop - Fcorao) * ((1 / Ccorao) + ((Px2 / (1ml)) * exp(Vcorao / Vx1) / (exp(Vcorao / Vx1) - 1) ^ 2)) + PplcFOL_dt
    Vaop_dt = (Paopc - (Vaop / Caop) + Px2 * (1 / (exp(Vaop / Vx8) - 1)) - PplcFOL) / Rtaop
    Vaod_dt = Faop - Faod - Fcrb
    Vsap_dt = Faod - Fsap
    Vsa_dt = Fsap - Fsa
    Vsc_dt = Fsa - Fsc
    Vsv_dt = Fsc - Fsv
    Vvc_dt = Fsv + Fcrb - Fvc
    Faop_dt = (Paopc - Faop * Raop - Paodc) / Laop
    Faod_dt = (Paodc - Faod * Raod - Psap) / Laod

    # pulmonary
    Vpap_dt = Frv - Fpap
    Vpad_dt = Fpap - Fpad
    Vpa_dt = Fpad - Fps - Fpa
    Vpc_dt = Fpa - Fpc
    Vpv_dt = Fpc + Fps - Fpv
    Fpap_dt = (Ppapc - Ppadc - Fpap * Rpap) / Lpa
    Fpad_dt = (Ppadc - Ppac - Fpad * Rpad) / Lpad

    # airway mechanics
    dV_dt = (Vcw - dV) / tau
    Vcw_dt = VA_dt + Vc_dt
    Qdotsm - Alvflux - VA_dt = 0
    VA_dt - Vve_dt - Qdotve = 0
    Vc_dt = Qdotco - Qdotsm

    # gas exchange
    PO2_ao_dt = (COutput * (CtO2_pa - CtO2_ao) + DL_O2 * (PA_O2 - PO2_ao)) / (Vpc * (alphaO2 + Hcrit * CHb * dSHbO2dPO2_ao))
    PO2_pa_dt = (COutput * (CtO2_ao - CtO2_pa) + PS * tweight * alphaO2 * (PO2_isf - PO2_pa)) / ((Vsc + Vcorcap) * (alphaO2 + Hcrit * CHb * dSHbO2dPO2_pa))
    PO2_isf_dt = PS * tweight * (PO2_pa - PO2_isf) / Visf - V_O2 / (alphaO2 * Visf)
    PCO2_ao_dt = (COutput * (CtCO2_pa - CtCO2_ao) + DL_CO2 * (PA_CO2 - PCO2_ao)) /(Vpc * (alphaCO2 + Hcrit * CHb * dSHbCO2dPCO2_ao))
    PCO2_pa_dt = (COutput * (CtCO2_ao - CtCO2_pa) + PS * tweight * alphaCO2 * (PCO2_isf - PCO2_pa)) /((Vsc + Vcorcap) * (alphaCO2 + Hcrit * CHb * dSHbCO2dPCO2_pa))
    PCO2_isf_dt = PS * tweight * (PCO2_pa - PCO2_isf) / Visf + V_CO2 / (alphaCO2 * Visf)
    if Qdotup > 0:
        PD_O2_dt = (Qdotup * Pao_O2 - Qdotco * PD_O2) / VD
    else:
        PD_O2_dt = (Qdotup*PD_O2 - Qdotco*PC_O2)/VD

    if Qdotup > 0 and Qdotsm > 0:
        PC_O2_dt = (Qdotco*PD_O2 - Qdotsm*PC_O2 - PC_O2*Vc_dt)/Vc
    elif Qdotup > 0 and Qdotsm < 0:
        PC_O2_dt = (Qdotco*PD_O2 - Qdotsm*PA_O2 - PC_O2*Vc_dt)/Vc
    elif Qdotup < 0 and Qdotsm > 0:
        PC_O2_dt = (Qdotco*PC_O2 - Qdotsm*PC_O2 - PC_O2*Vc_dt)/Vc
    else:
        PC_O2_dt = (Qdotco*PC_O2 - Qdotsm*PA_O2 - PC_O2*Vc_dt)/Vc

    if Qdotsm > 0:
        PA_O2_dt = (Qdotsm*PC_O2 - (Pstp*Tbody/Tstp)*O2flux - VA_dt*PA_O2)/VA
    else:
        PA_O2_dt = (Qdotsm*PA_O2 - (Pstp*Tbody/Tstp)*O2flux - VA_dt*PA_O2)/VA

    if Qdotup>0:
        PD_CO2_dt = (Qdotup*Pao_CO2 - Qdotco*PD_CO2)/VD
    else :
        PD_CO2_dt = (Qdotup*PD_CO2 - Qdotco*PC_CO2)/VD

    if Qdotup > 0 and Qdotsm > 0:
        PC_CO2_dt = (Qdotco*PD_CO2 - Qdotsm*PC_CO2 - PC_CO2*Vc_dt)/Vc
    elif Qdotup > 0 and Qdotsm < 0:
        PC_CO2_dt = (Qdotco*PD_CO2 - Qdotsm*PA_CO2 - PC_CO2*Vc_dt)/Vc
    elif Qdotup < 0 and Qdotsm > 0:
        PC_CO2_dt = (Qdotco*PC_CO2 - Qdotsm*PC_CO2 - PC_CO2*Vc_dt)/Vc
    else:
        PC_CO2_dt = (Qdotco*PC_CO2 - Qdotsm*PA_CO2 - PC_CO2*Vc_dt)/Vc

    if Qdotsm>0:
        PA_CO2_dt = (Qdotsm*PC_CO2 -(Pstp*Tbody/Tstp)*(CO2flux) - PA_CO2*VA_dt)/VA
    else:
        PA_CO2_dt = (Qdotsm*PA_CO2 -(Pstp*Tbody/Tstp)*(CO2flux) - PA_CO2*VA_dt)/VA

    # chemoreceptors
    gc_dt = (fc - gc) / tauv

    # blood gas handling
    PBC_pc_dt = (Fpc / Vpc) * (PBC_sc - PBC_pc) + kp5 * CtCO2_ao * (Cheme - PBC_pc) * (SHbO2_ao / (1 + H_ao / K3bgh) + (1 - SHbO2_ao) / ( 1 + H_ao / K2bgh)) - kp5 * PBC_pc * H_ao * ( SHbO2_ao / K6bgh + (1 - SHbO2_ao) / K5bgh)
    PBC_sc_dt = (COutput / (Vsc + Vcorcap)) * (PBC_pc - PBC_sc) + kp5 * CtCO2_pa * (Cheme - PBC_sc) * ( SHbO2_pa / (1 + H_pa / K3bgh) + (1 - SHbO2_pa) / (1 + H_pa / K2bgh)) - kp5 * PBC_sc * H_pa * ( SHbO2_pa / K6bgh + (1 - SHbO2_pa) / K5bgh)
    HCO3_ao_dt = (Fpc / Vpc) * (HCO3_pa - HCO3_ao) + CFb * (kp1 * CtCO2_ao - km1 * HCO3_ao * H_ao / K1bgh)
    HCO3_pa_dt = (COutput / (Vsc + Vcorcap)) * (HCO3_ao - HCO3_pa) + CFb * ( kp1 * CtCO2_pa - km1 * HCO3_pa * H_pa / K1bgh)
    HCO3_isf_dt = (PS_HCO3 * tweight / Visf) * (HCO3_ao - HCO3_isf) + CFt * ( kp1 * CtCO2_isf - km1 * HCO3_isf * H_isf / K1bgh)
    if H_ao > 4E-8:
        Hout_dt = (H_ao - 4E-8)/tauHout
    else:
        Hout_dt = 0
    H_ao_dt = (Fpc / Vpc) * (H_pa - H_ao) + (2.303 / Betab) * H_ao * (CFb * (kp1 * CtCO2_ao - km1 * H_ao * HCO3_ao / K1bgh) + 1.5 * ( PBC_pc_dt) - 0.6 * alphaO2 * DVRO11 * ( PO2_ao_dt)) -Hout_dt
    H_pa_dt = (COutput / (Vsc + Vcorcap)) * (H_ao - H_pa) + (2.303 / Betab) * H_pa * (CFb * (kp1 * CtCO2_pa - km1 * H_pa * HCO3_pa / K1bgh) + 1.5 * ( PBC_sc_dt) - 0.6 * alphaO2 * DVRO21 * ( PO2_pa_dt))
    H_isf_dt = (PS_H * tweight / Visf) * (H_ao - H_isf) + (2.303 / Betat) * H_isf * CFt * (kp1 * CtCO2_isf - km1 * H_isf * HCO3_isf / K1bgh) + (facid * GO2isf * tweight * CtO2_isf / Visf)

    # coronary circulation
    Vcorao_dt = Flv - Vaop_dt - Faop - Fcorao
    Vcorea_dt = Fcorao - Fcorea
    Vcorla_dt = Fcorea - Fcorla
    Vcorsa_dt = Fcorla - Fcorsa
    Vcorcap_dt = Fcorsa - Fcorcap
    Vcorsv_dt = Fcorcap - Fcorsv
    Vcorlv_dt = Fcorsv - Fcorlv
    Vcorev_dt = Fcorlv - Fcorev



    # total fluid
    Vtot = Vra + Vrv + Vpap + Vpad + Vpa + Vpc + Vpv + Vla + Vlv + Vaop + Vaod +Vsa + Vsap + Vsc + Vsv + Vvc + perifl + Vcorcirc
    VBcirc = Vtot - perifl
    SysArtVol = Vaop + Vaod + Vsa + Vsap
    SysVenVol = Vsv + Vvc
    PulArtVol = Vpap + Vpad + Vpa
    PulVenVol = Vpv


    # d_Nbr_dt = Nbr_t
    # d_Nbr_t_dt = _firing_frequency(d_ABPfol_dt, Nbr_t, Nbr, ABPshift, a, a1, a2, K)
    # d_N_con_dt = _n_change(t, tHB[0], l_con, N_con, K_con, Nbr_list, Nbr_list_idx, T_con)
    # d_N_vaso_dt = _n_change(t, tHB[0], l_vaso, N_vaso, K_vaso, Nbr_list, Nbr_list_idx, T_vaso)

    dy = np.array([
        d_Vra_dt, d_Vrv_dt, d_Vla_dt, d_Vlv_dt,
        d_MAPmeas_dt, d_Faop_dt, d_Faod_dt, d_Frv_sm_dt, d_Vaop_dt, d_Vaod_dt, d_Vsa_dt, d_Vsap_dt, d_Vsc_dt, d_Vsv_dt,
        d_Vvc_dt, d_Paop_dt, d_AOFmod_dt, d_ABPfol_dt, d_COmea_dt,
        d_Vpap_dt, d_Vpad_dt, d_Vpa_dt, d_Vpc_dt, d_Vpv_dt, d_Fpap_dt, d_Fpad_dt,
        d_Vcorepi_dt, d_Vcorintra_dt, d_Vcorcap_dt, d_Vcorvn_dt,
        d_Nbr_dt, d_Nbr_t_dt, d_N_con_dt, d_N_vaso_dt,
    ])

    # print(f"\t TBV[{TBV:.2f}]")
    # print(f"\t d_Vra_dt[{d_Vra_dt:.2f}] d_Vrv_dt[{d_Vrv_dt:.2f}] d_Vla_dt[{d_Vla_dt:.2f}] d_Vlv_dt[{d_Vlv_dt:.2f}]")

    # save some updated variables
    if np.min(np.abs(t - tmeas)) < 1E-4:
        Nbr_list.append(Nbr)
        Nbr_list_idx.append(t)

        t_list.append(t)

        Pra_list.append(Pra)
        Prv_list.append(Prv)
        Pla_list.append(Pla)
        Plv_list.append(Plv)

        Vra_list.append(Vra)
        Vrv_list.append(Vrv)
        Vla_list.append(Vla)
        Vlv_list.append(Vlv)

    change_list.append([HRa, HRv, Tsv, Tsa, Vvarlvs0, Vvarrvs0, n, m, tRwave, tPwave, af_con2])

    return dy


def call_cardio(args, params, debug=False):
    # load input data
    t = np.arange(0, 20, 0.01)
    mlSTPD = 1 / 22400

    # heart
    Vra = 58.9
    Vrv = 109.47
    Vla = 78.8
    Vlv = 120.46
    EDPLV = 5
    EDPRV = 2.5
    EDVLV = 126
    EDVRV = 176
    # systemic circulation
    MAP = 93
    Faop = 10.2
    Faod = 53.63
    COutput = 96.6
    Vaop = 64.6
    Vaod = 209.75
    Vsa = 226.2
    Vsap = 175.5
    Vsc = 315.8
    Vsv = 3087.32
    Vvc = 200.65
    PaodFOL = 93
    # pulmonary circulation
    Vpap = 18.5
    Vpad = 30.85
    Vpa = 68.74
    Vpc = 114.7
    Vpv = 201.55
    Fpap = 405.3
    Fpad = 65.7
    Paopc = 93
    # baroreceptor
    tshift = t[0]
    HR = 77
    Tsv = params.loc["TsvK"] * np.sqrt(1.0 / HR / 60)
    Tsa = params.loc["TsaK"] * np.sqrt(1.0 / HR / 60)
    Nbr = 141.8
    Nbr_t = 1044.0
    N_hrv = 112.8
    N_hrs = 110.1
    N_con = 129.2
    N_vaso = 129.5
    afs_con2 = 1
    # chemoreceptors
    gc = 0.008
    # airway mechanics
    Vc = 0.08
    VA = 3.3
    Vve = -0.08
    Vcw = VA + Vc + params.loc["VD"]
    dV = Vcw
    tresp = t[0]
    A = params.loc["KA"] * gc + params.loc["DA"]
    B = params.loc["KB"] * gc + params.loc["DB"]
    C = params.loc["KC"] * gc + params.loc["DC"]
    if C / (2 * np.pi) > (120):
        RespRcont = 120
    else:
        RespRcont = C / (2 * np.pi)
    RespR = RespRcont
    Pcw = (A * np.sin(2 * np.pi * RespR * (t - tresp)) - B)
    PplcFOL = Pcw
    # gas exchange
    PO2_isf = 19.4
    PO2_ao = 89.7
    PO2_pa = 34.9
    PCO2_ao = 41.9
    PCO2_pa = 60.9
    PCO2_isf = 61.4
    PD_O2 = 142.4
    PC_O2 = 146.4
    PA_O2 = 100.7
    PD_CO2 = 1.6
    PC_CO2 = 2.8
    PA_CO2 = 41.2
    # blood gas handling
    PBC_pc = 0.0060761
    PBC_sc = 0.00658468
    HCO3_ao = 0.030934
    HCO3_pa = 0.038
    HCO3_isf = 0.0205
    H_ao = 4e-8
    H_pa = 7e-8
    H_isf = 7.2e-8
    Hout = 2.0e-5
    # coronary circulation
    Vcorao = 4.8379313859
    Vcorea = 2.4
    Vcorla = 2.3
    Vcorsa = 2.7
    Vcorcap = 11.4
    Vcorsv = 4
    Vcorlv = 3.1
    Vcorev = 1.1

    # initial condition for the ODE solver
    y0 = np.array([
        Vra, Vrv, Vla, Vlv,
        MAPmeas, Faop, Faod, Frv_sm, Vaop, Vaod, Vsa, Vsap, Vsc, Vsv, Vvc, Paop, AOFmod, ABPfol, COmea,
        Vpap, Vpad, Vpa, Vpc, Vpv, Fpap, Fpad,
        Vcorepi, Vcorintra, Vcorcap, Vcorvn,
        Nbr, Nbr_t, N_con, N_vaso,
    ])

    Nbr_list = []
    Nbr_list_idx = []

    # # age factor
    # tau = 100
    # t = np.linspace(0, 100, 100)
    # # x = 1 - t / tau * np.exp(-t / tau / 2) + t / tau * np.exp(-t / tau)
    # # x = 2 / (1 + np.exp(-1 / 8 * (t - 60)))
    # x = 1 / (1 + np.exp(1 / 10 * (t - 80)))
    #
    # age_factor = np.interp(args.age, t, x)
    # angiotensin_factor = 0.9
    # factor = angiotensin_factor * age_factor
    #
    # params.loc["Caop", "value"] = params.loc["Caop", "value"] * factor
    # params.loc["Caod", "value"] = params.loc["Caod", "value"] * factor
    # params.loc["Csap", "value"] = params.loc["Csap", "value"] * factor
    # params.loc["Csc", "value"] = params.loc["Csc", "value"] * factor
    # params.loc["Cpap", "value"] = params.loc["Cpap", "value"] * factor
    # params.loc["Cpad", "value"] = params.loc["Cpad", "value"] * factor
    # params.loc["Cpa", "value"] = params.loc["Cpa", "value"] * factor
    # params.loc["Cpc", "value"] = params.loc["Cpc", "value"] * factor
    # params.loc["Cpv", "value"] = params.loc["Cpv", "value"] * factor
    # params.loc["Ccorepi", "value"] = params.loc["Ccorepi", "value"] * factor
    # params.loc["Ccorintra", "value"] = params.loc["Ccorintra", "value"] * factor
    # params.loc["Ccorcap", "value"] = params.loc["Ccorcap", "value"] * factor
    # params.loc["Ccorvn", "value"] = params.loc["Ccorvn", "value"] * factor
    # params.loc["KElv", "value"] = params.loc["KElv", "value"] * factor
    # params.loc["KErv", "value"] = params.loc["KErv", "value"] * factor
    # params.loc["Emaxlv1", "value"] = params.loc["Emaxlv1", "value"] / factor
    # params.loc["Eminlv", "value"] = params.loc["Eminlv", "value"] / factor
    # params.loc["Emaxrv1", "value"] = params.loc["Emaxrv1", "value"] / factor
    # params.loc["Eminrv", "value"] = params.loc["Eminrv", "value"] / factor
    # params.loc["Emaxra", "value"] = params.loc["Emaxra", "value"] / factor
    # params.loc["Eminra", "value"] = params.loc["Eminra", "value"] / factor
    # params.loc["Emaxla", "value"] = params.loc["Emaxla", "value"] / factor
    # params.loc["Eminla", "value"] = params.loc["Eminla", "value"] / factor

    t_list = []
    Pra_list, Prv_list, Pla_list, Plv_list = [], [], [], []
    Vra_list, Vrv_list, Vla_list, Vlv_list = [], [], [], []

    ODE_args = (
        x, tHB, HP, tmeas, ABPmeas, PAFmeas, Nbr_list, Nbr_list_idx,
        [[], [HRa, HRv, Tsv, Tsa, Vvarlvs0, Vvarrvs0, n, m, tRwave, tPwave, af_con2]],
        t_list,
        Pra_list, Prv_list, Pla_list, Plv_list,
        Vra_list, Vrv_list, Vla_list, Vlv_list,

        params.loc["Ts1v", "value"],
        params.loc["Ts1a", "value"],
        params.loc["Ts2", "value"],
        params.loc["offv", "value"],
        params.loc["Vlvd0", "value"],
        params.loc["Vlvs0", "value"],
        params.loc["Vrvd0", "value"],
        params.loc["Vrvs0", "value"],
        params.loc["Vlad0", "value"],
        params.loc["Vlas0", "value"],
        params.loc["Vrad0", "value"],
        params.loc["Vras0", "value"],
        params.loc["Rra", "value"],
        params.loc["Rla", "value"],
        params.loc["Rlv", "value"],
        params.loc["Rrv", "value"],
        params.loc["PRint", "value"],
        params.loc["KElv", "value"],
        params.loc["KErv", "value"],
        params.loc["Emaxlv1", "value"],
        params.loc["Eminlv", "value"],
        params.loc["Emaxrv1", "value"],
        params.loc["Eminrv", "value"],
        params.loc["EDVLV", "value"],
        params.loc["EDVRV", "value"],
        params.loc["Emaxra", "value"],
        params.loc["Eminra", "value"],
        params.loc["Emaxla", "value"],
        params.loc["Eminla", "value"],

        params.loc["KCOMAP", "value"],
        params.loc["Raop", "value"],
        params.loc["Rtaop", "value"],
        params.loc["Rcrb", "value"],
        params.loc["Raod", "value"],
        params.loc["Rtaod", "value"],
        params.loc["Rsap", "value"],
        params.loc["Rsc", "value"],
        params.loc["Rsv", "value"],
        params.loc["Caop", "value"],
        params.loc["Caod", "value"],
        params.loc["Csap", "value"],
        params.loc["Csc", "value"],
        params.loc["Vaop0", "value"],
        params.loc["Vaod0", "value"],
        params.loc["Vsap0", "value"],
        params.loc["Vsc0", "value"],
        params.loc["Laop", "value"],
        params.loc["Laod", "value"],
        params.loc["Kc", "value"],
        params.loc["Do", "value"],
        params.loc["Vsa0", "value"],
        params.loc["Vsa_max", "value"],
        params.loc["Kp1", "value"],
        params.loc["Kp2", "value"],
        params.loc["Kr", "value"],
        params.loc["Rsa0", "value"],
        params.loc["tau_p", "value"],
        params.loc["Ksv", "value"],
        params.loc["Kv1", "value"],
        params.loc["Vmax_sv", "value"],
        params.loc["D2", "value"],
        params.loc["K1", "value"],
        params.loc["K2", "value"],
        params.loc["KR", "value"],
        params.loc["R0", "value"],
        params.loc["Vvc0", "value"],
        params.loc["Vmax_vc", "value"],
        params.loc["Vmin_vc", "value"],
        params.loc["tauCO", "value"],
        params.loc["Kxp", "value"],
        params.loc["Kxv", "value"],
        params.loc["Kxv1", "value"],
        params.loc["Kxp1", "value"],
        params.loc["tauMAP", "value"],
        params.loc["tauABP", "value"],

        params.loc["Rtpap", "value"],
        params.loc["Rtpad", "value"],
        params.loc["Rpap", "value"],
        params.loc["Rpad", "value"],
        params.loc["Rps", "value"],
        params.loc["Rpa", "value"],
        params.loc["Rpc", "value"],
        params.loc["Rpv", "value"],
        params.loc["Cpap", "value"],
        params.loc["Cpad", "value"],
        params.loc["Cpa", "value"],
        params.loc["Cpc", "value"],
        params.loc["Cpv", "value"],
        params.loc["Vpap0", "value"],
        params.loc["Vpad0", "value"],
        params.loc["Vpa0", "value"],
        params.loc["Vpc0", "value"],
        params.loc["Vpv0", "value"],
        params.loc["Lpap", "value"],
        params.loc["Lpad", "value"],

        params.loc["Rcorepi", "value"],
        params.loc["Rcorintra", "value"],
        params.loc["Rcorcap", "value"],
        params.loc["Rcorvn", "value"],
        params.loc["Ccorepi", "value"],
        params.loc["Ccorintra", "value"],
        params.loc["Ccorcap", "value"],
        params.loc["Ccorvn", "value"],
        params.loc["Vcorepi0", "value"],
        params.loc["Vcorintra0", "value"],
        params.loc["Vcorcap0", "value"],
        params.loc["Vcorvn0", "value"],

        params.loc["a", "value"],
        params.loc["a1", "value"],
        params.loc["a2", "value"],
        params.loc["K", "value"],
        params.loc["K_con", "value"],
        params.loc["T_con", "value"],
        params.loc["l_con", "value"],
        params.loc["a_con", "value"],
        params.loc["b_con", "value"],
        params.loc["tau_con", "value"],
        params.loc["No_con", "value"],
        params.loc["K_vaso", "value"],
        params.loc["T_vaso", "value"],
        params.loc["l_vaso", "value"],
        params.loc["a_vaso", "value"],
        params.loc["tau_vaso", "value"],
        params.loc["No_vaso", "value"],
        params.loc["amin", "value"],
        params.loc["bmin", "value"],
        params.loc["Ka", "value"],
        params.loc["Kb", "value"]
    )

    if not debug:
        max_time_step = 10
        sol = solve_ivp(fun=ODE,
                        t_span=[tHB[0], tHB[max_time_step]],
                        y0=y0,
                        args=ODE_args,
                        t_eval=tm[tm < tHB[max_time_step]],
                        # first_step=0.02,
                        # rtol=1e-1, atol=1e-2,
                        method="LSODA")
        cols = [
            "t",
            "Vra", "Vrv", "Vla", "Vlv",
            "MAPmeas", "Faop", "Faod", "Frv_sm", "Vaop", "Vaod", "Vsa", "Vsap", "Vsc", "Vsv", "Vvc", "Paop", "AOFmod",
            "ABPfol", "COmea",
            "Vpap", "Vpad", "Vpa", "Vpc", "Vpv", "Fpap", "Fpad",
            "Vcorepi", "Vcorintra", "Vcorcap", "Vcorvn",
            "Nbr", "Nbr_t", "N_con", "N_vaso"
        ]
        t_df = pd.DataFrame(sol["t"])
        y_df = pd.DataFrame(sol["y"].T)
        y_df = pd.concat([t_df, y_df], axis=1)
        y_df.columns = cols
        y_df.to_csv("cardio_y.csv")

        pv_lists = [t_list, Pra_list, Prv_list, Pla_list, Plv_list, Vra_list, Vrv_list, Vla_list, Vlv_list]
        pv_cols = ["t", "Pra", "Prv", "Pla", "Plv", "Vra", "Vrv", "Vla", "Vlv"]
        pv_df = pd.DataFrame.from_records(pv_lists).T
        pv_df.columns = pv_cols

        output_dir = f"data/drug-{args.drug_name}_glu-{args.glu}_infection-{int(args.infection)}_renal-{args.renal_function}_age-{args.age}"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        pv_df.to_csv(os.path.join(output_dir, "pv.csv"))

    return
