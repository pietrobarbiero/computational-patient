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

        t_meas, dt, mlSTPD,
        params_list,

        Vlvrd,
        Vlvrs,
        Vrvrd,
        Vrvrs,
        Vlard,
        Vlars,
        Vrard,
        Vrars,
        Rra,
        Rla,
        PRint,
        Emaxlv,
        Eminlv,
        Emaxrv,
        Eminrv,
        Emaxra,
        Eminra,
        Emaxla,
        Eminla,
        Pbs,
        Vmyo,
        Punit,
        TsvK,
        TsaK,
        Rav,
        Raop,
        Rcrb,
        Raod,
        Rtaop,
        Rtaod,
        Rsap,
        Rsc,
        Rsv,
        Caop,
        Caod,
        Csap,
        Csc,
        Laop,
        Laod,
        Kc,
        Do,
        Vsa_o,
        Vsa_max,
        Kp1,
        Kp2,
        Kr,
        Rsao,
        tau_p,
        Kv,
        Vmax_sv,
        D2,
        K1,
        K2,
        KR,
        Ro,
        Vo,
        Vmax_vc,
        Vmin_vc,
        COtau,
        Px2,
        Vx8,
        Vx75,
        Vx1,
        Px1,
        Rpuv,
        Rtpap,
        Rtpad,
        Rpap,
        Rpad,
        Rps,
        Rpa,
        Rpc,
        Rpv,
        Ctpap,
        Ctpad,
        Cpa,
        Cpc,
        Cpv,
        Lpa,
        Lpad,
        a,
        a1,
        a2,
        K,
        K_hrv,
        T_hrv,
        L_hrv,
        a_hrv,
        tau_hrv,
        No_hrv,
        K_hrs,
        T_hrs,
        L_hrs,
        a_hrs,
        tau_hrs,
        No_hrs,
        K_con,
        T_con,
        L_con,
        a_con,
        b_con,
        tau_con,
        No_con,
        K_vaso,
        T_vaso,
        L_vaso,
        a_vaso,
        tau_vaso,
        No_vaso,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        amin,
        bmin,
        Ka,
        Kb,
        RV,
        VD,
        Ac,
        Al,
        As,
        Au,
        Bc,
        Bcp,
        Bl,
        Bs,
        Cve,
        Kc_air,
        Kl,
        Ks,
        Ku,
        Rve,
        Vstar,
        Vcmax,
        tau,
        KA,
        KB,
        KC,
        DA,
        DB,
        DC,
        Ccw,
        Tbody,
        Pstp,
        Tstp,
        nH,
        P50_O2,
        CHb,
        Hcrit,
        alphaO2,
        Visf,
        PS,
        Vcytox,
        Kcytox,
        P50_CO2,
        alphaCO2,
        RQ,
        PH2O,
        Vpcmax,
        Pao,
        r_Pao_O2,
        r_Pao_CO2,
        tauv,
        Pche,
        kp1,
        km1,
        K1bgh,
        K2bgh,
        K3bgh,
        K5bgh,
        K6bgh,
        kp5,
        CFb,
        CFt,
        Betab,
        Betat,
        O2cap,
        Hbconc,
        Cheme,
        HpNorm,
        facid,
        tweight,
        tauHout,
        PS_H,
        PS_HCO3,
        K_pcd,
        phi_pcd,
        Vpcd_o,
        perifl,
        Rcorao,
        Rcorea,
        Rcorla,
        Rcorsa,
        Rcorcap,
        Rcorsv,
        Rcorlv,
        Rcorev,
        Ccorao,
        Ccorea,
        Ccorla,
        Ccorsa,
        Ccorcap,
        Ccorsv,
        Ccorlv,
        Ccorev,
        ):

    Vla, \
    Vlv, \
    Vra, \
    Vrv, \
    COutput, \
    Paopc, \
    MAP, \
    Vaop, \
    Vaod, \
    Vsap, \
    Vsa, \
    Vsc, \
    Vsv, \
    Vvc, \
    Faop, \
    Faod, \
    Vpap, \
    Vpad, \
    Vpa, \
    Vpc, \
    Vpv, \
    Fpap, \
    Fpad, \
    dV, \
    PaodFOL, \
    PplcFOL, \
    Vcw, \
    VA, \
    Vve, \
    Vc, \
    PO2_ao, \
    PO2_pa, \
    PO2_isf, \
    PCO2_ao, \
    PCO2_pa, \
    PCO2_isf, \
    PD_O2, \
    PC_O2, \
    PA_O2, \
    PD_CO2, \
    PC_CO2, \
    PA_CO2, \
    gc, \
    PBC_pc, \
    PBC_sc, \
    HCO3_ao, \
    HCO3_pa, \
    HCO3_isf, \
    Hout, \
    H_ao, \
    H_pa, \
    H_isf, \
    Vcorao, \
    Vcorea, \
    Vcorla, \
    Vcorsa, \
    Vcorcap, \
    Vcorsv, \
    Vcorlv, \
    Vcorev, \
    Nbr, \
    Nbr_t, \
    N_hrv, \
    N_hrs, \
    N_con, \
    N_vaso = y

    tshift_prev, HR_prev, HRcont_prev, afs_con_prev, ylv_prev, \
    Plvc_prev, Prvc_prev, Vlv_prev, Vrv_prev,\
    Tsa_prev, Tsv_prev, afs_con2_prev,\
    RespR_prev, tresp_prev = params_list

    tshift = tshift_prev
    Tsa = Tsa_prev
    Tsv = Tsv_prev
    afs_con2 = afs_con2_prev
    RespR = RespR_prev
    tresp = tresp_prev
    HR = HR_prev
    dt = dt

    # triggers
    if t + dt >= tshift_prev + 1/HR_prev:
        Tsv = TsvK * np.sqrt(1 / HR_prev / 60)
        Tsa = TsaK * np.sqrt(1 / HR_prev / 60)
        HR = HRcont_prev
        tshift = t
        afs_con2 = afs_con_prev

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

    if ylv_prev == 0 and ylv > 0:
        EDPLV = Plvc_prev
        EDPRV = Prvc_prev
        EDVLV = Vlv_prev
        EDVRV = Vrv_prev

    if t>tshift and t<tshift + 0.02:
        P_QRSwave = 5
    elif t>(tshift+PRint) and t<(tshift+PRint+0.02):
        P_QRSwave = 10
    else:
        P_QRSwave = 0

    # ----------------- baroreceptor ----------------- #

    b_hrv = 1 - a_hrv
    F_hrv = a_hrv + (b_hrv / (np.exp(tau_hrv * (N_hrv - No_hrv)) + 1.0))
    b_hrs = 1 - a_hrs
    F_hrs = a_hrs + (b_hrs / (np.exp(tau_hrs * (N_hrs - No_hrs)) + 1.0))
    F_con = a_con + (b_con / (np.exp(tau_con * (N_con - No_con)) + 1.0))
    b_vaso = 1 - a_vaso
    F_vaso = a_vaso + (b_vaso / (np.exp(tau_vaso * (N_vaso - No_vaso)) + 1.0))
    HRcont = (h1 + (h2 * F_hrs) - (h3 * F_hrs ** 2) - (h4 * F_hrv) + (h5 * F_hrv ** 2) - (h6 * F_hrv * F_hrs))
    afs_con = amin + (Ka * F_con)
    bfs_con = bmin + (Kb * F_con)


    # ----------------- heart ----------------- #

    Vcorcirc = Vcorao + Vcorea + Vcorla + Vcorsa + Vcorcap + Vcorsv + Vcorlv + Vcorev

    # airway mechanics
    A = KA * gc + DA
    B = KB * gc + DB
    C = KC * gc + DC
    if C/(2*np.pi)>120: #(120min**-1):
        RespRcont = 120 #120min**-1
    else:
        RespRcont = C / (2 * np.pi)
    if t>=tresp+(1/RespR_prev):
        RespR = RespRcont
        tresp = t
    Pl = Al * np.exp(Kl * VA) + Bl
    Pcw = (A * np.sin(2 * np.pi * RespR * (t - tresp)) - B)
    if Vc / Vcmax < 0.5:
        Pc = Ac - Bc * (Vc / Vcmax - 0.7) ** 2
    else:
        Pc = Ac - Bc * (0.5 - 0.7) ** 2 - Bcp * np.log(Vcmax / Vc - 0.999)
    Ppl = Pcw
    Pve = Vve / Cve
    PplmmHg = Ppl
    Pmouth = Pbs
    Pplc = Ppl

    # pericardium
    Vpcd = Vrv + Vra + Vlv + Vla + perifl + Vmyo + Vcorcirc
    Ppcd = (K_pcd * np.exp((Vpcd - Vpcd_o) / phi_pcd)) - Px2 * (1 / (np.exp(Vpcd / Vx75) - 1))
    Ppcdc = Ppcd + Pplc

    Era= (Emaxra-Eminra)*yra+Eminra
    Vrar= (1-yra)*(Vrard-Vrars) + Vrars
    Erv = ((Emaxrv - Eminrv) * yrv) + Eminrv
    Vrvr = (1 - yrv) * (Vrvrd - Vrvrs) + Vrvrs
    Ela = (Emaxla - Eminla) * yla + Eminla
    Vlar = (1 - yla) * (Vlard - Vlars) + Vlars
    Elv = ((Emaxlv - Eminlv) * ylv) + Eminlv
    Vlvr = (1 - ylv) * (Vlvrd - Vlvrs) + Vlvrs
    Pla = (Vla - Vlar) * Ela - Px2 * (1 / (np.exp((Vla) / Vx8) - 1))
    Plac = Pla + Ppcdc
    Plv = afs_con2 * (Vlv - Vlvr) * Elv - Px2 * (1 / (np.exp((Vlv) / Vx8) - 1))
    Plvc = Plv + Ppcdc
    Pra = (Vra - Vrar) * Era - Px2 * (1 / (np.exp((Vra) / Vx8) - 1))
    Prac = Pra + Ppcdc
    Prv = afs_con2 * (Vrv - Vrvr) * Erv - Px2 * (1 / (np.exp(Vrv / Vx8) - 1))
    Prvc = Prv + Ppcdc

    # pulmonary
    Ppapc1 = (Rpuv * (Vpap / Ctpap) - Rpuv * (Px2 * (1 / (np.exp(Vpap / Vx8) - 1))) + Prvc * Rtpap - Rpuv * Rtpap * Fpap + Pplc * Rpuv) / (Rtpap + Rpuv)
    Ppapc2 = (Vpap / Ctpap + Pplc - Rtpap * Fpap - Px2 * (1 / (np.exp(Vpap / Vx8) - 1)))
    if Prvc>Ppapc1:
        Ppapc = Ppapc1
    else:
        Ppapc = Ppapc2
    # pulmonary

    if Plac > Plvc:
        Fla = (Plac - Plvc) / Rla
    else:
        Fla = 0
    if Plvc>Paopc:
        Flv = (Plvc-Paopc)/Rav
    else:
        Flv = 0
    if Prac>Prvc:
        Fra = (Prac-Prvc)/Rra
    else:
        Fra = 0
    if Prvc>Ppapc:
        Frv = (Prvc - Ppapc) / Rpuv
    else:
        Frv = 0
    SV = COutput / HR


    # ----------------- systemic ----------------- #

    Rsa = Rsao + (Kr * np.exp(4 * F_vaso)) + (Kr * (Vsa_max / Vsa) ** 2)
    Rvc = (KR * (Vmax_vc / Vvc) ** 2) + Ro
    Paop = Paopc - PplcFOL
    Psa_a = Kc * np.log(((Vsa - Vsa_o) / Do) + 1)
    Psa_p = Kp1 * np.exp(tau_p * (Vsa - Vsa_o)) + Kp2 * (Vsa - Vsa_o) ** 2
    Psa = F_vaso * Psa_a + (1 - F_vaso) * Psa_p
    Psap = Vsap / Csap - Px2 * (1 / (np.exp(Vsap / Vx8) - 1))
    Psc = Vsc / Csc - Px2 * (1 / (np.exp(Vsc / Vx8) - 1))
    Psv = -Kv * np.log((Vmax_sv / Vsv) - 0.99)
    if Vvc>Vo:
        Pvc = D2+K2*np.exp(Vo/Vmin_vc)+K1*(Vvc-Vo) - Px2/(np.exp(Vvc/Vx8)-1)
    else:
        Pvc = D2+K2*np.exp(Vvc/Vmin_vc) - Px2/(np.exp(Vvc/Vx8)-1)
    Pvcc = Pvc + Pplc
    Paodc = (Rtaod * Rcrb * Faop - Rtaod * Rcrb * Faod + (Vaod * Rcrb / Caod) - Rcrb * Px2 / (np.exp(Vaod / Vx8) - 1) + Pvcc * Rtaod) / (Rcrb + Rtaod)
    Paod = Paodc - Pbs
    Fcrb = (Paodc - Pvcc) / Rcrb
    Fsap = (Psap - Psa) / Rsap
    Fsa = (Psa - Psc) / Rsa
    Fsc = (Psc - Psv) / Rsc
    Fsv = (Psv - Pvcc) / Rsv
    Fvc = (Pvcc - Prac) / Rvc


    # ----------------- pulmonary ----------------- #
    # Ppapc1 = (Rpuv * (Vpap / Ctpap) - Rpuv * (Px2 * (1 / (exp(Vpap / Vx8) - 1))) + Prvc * Rtpap - Rpuv * Rtpap * Fpap + Pplc * Rpuv) / (Rtpap + Rpuv)
    # Ppapc2 = (Vpap / Ctpap + Pplc - Rtpap * Fpap - Px2 * (1 / (exp(Vpap / Vx8) - 1)))
    # if Prvc>Ppapc1:
    #     Ppapc = Ppapc1
    # else:
    #     Ppapc = Ppapc2

    Vpad_dt = Fpap - Fpad

    Ppap = Ppapc - Pplc
    Ppadc = Vpad_dt * Rtpad + Pplc + Vpad / Ctpad - Px2 * (1 / (np.exp(Vpad / Vx8) - 1))
    Ppad = Ppadc - Pplc
    Ppa = Vpa / Cpa - Px2 * (1 / (np.exp(Vpa / Vx8) - 1))
    Ppac = Ppa + Pplc
    Ppc = Vpc / Cpc - Px2 * (1 / (np.exp(Vpc / Vx8) - 1))
    Ppcc = Ppc + Pplc
    Ppv = Vpv / Cpv - Px2 * (1 / (np.exp(Vpv / Vx8) - 1))
    Ppvc = Ppv + Pplc
    Fps = (Ppac - Ppvc) / Rps
    Fpa = (Ppac - Ppcc) / Rpa
    Fpc = (Ppcc - Ppvc) / Rpc
    Fpv = (Ppvc - Plac) / Rpv


    # ----------------- airway mechanics ----------------- #
    # if t>=tresp+(1/RespR):
    #     RespR = RespRcont
    #     tresp = t

    dV_dt = (Vcw - dV) / tau

    Rco = 0
    if Vc > Vcmax:
        Rc = Kc_air + Rco
    else:
        Rc = Kc_air * (Vcmax / Vc) ** 2 + Rco
    Rs = As * np.exp(Ks * (VA - RV) / (Vstar - RV)) + Bs
    Ru = Au + Ku * abs(dV_dt)
    # A = KA * gc + DA
    # B = KB * gc + DB
    # C = KC * gc + DC
    # Pl = Al * exp(Kl * VA) + Bl
    # Pcw = (A * sin(2 * PI * RespR * (t - tresp)) - B)
    # if Vc / Vcmax < 0.5:
    #     Pc = Ac-Bc*(Vc/Vcmax - 0.7)**2
    # else:
    #     Pc = Ac-Bc*(0.5 - 0.7)**2-Bcp*ln(Vcmax/Vc - 0.999)
    # Ppl = Pcw
    # Pve = Vve / Cve
    # PplmmHg = Ppl
    # Pmouth = Pbs
    # Pplc = Ppl
    # if C/(2*PI)>(120min**-1):
    #     RespRcont = 120min**-1
    # else:
    #     C / (2 * PI)
    Pcc = Pplc + Pc
    PA = Ppl + Pve + Pl
    Ps = Pcc - PA
    Qdotco = (Pmouth - Pcc) / (Ru + Rc)
    Qdotup = Qdotco
    Qdotve = Pve / Rve
    Qdotsm = Ps / Rs
    PD = Qdotup * Ru
    PDc = Qdotco * Rc + Pcc
    Pup = Qdotup * (Ru + Rc)


    # ----------------- gas exchange ----------------- #
    SHbO2_ao = (PO2_ao / P50_O2) ** nH / (((PO2_ao / P50_O2) ** nH) + 1)
    SHbO2_pa = (PO2_pa / P50_O2) ** nH / (((PO2_pa / P50_O2) ** nH) + 1)
    dSHbO2dPO2_ao = nH * (PO2_ao ** -1) * (PO2_ao / P50_O2) ** nH / (((PO2_ao / P50_O2) ** nH) + 1) ** 2
    dSHbO2dPO2_pa = nH * (PO2_pa ** -1) * (PO2_pa / P50_O2) ** nH / (((PO2_pa / P50_O2) ** nH) + 1) ** 2
    SHbO2_aoISR = SHbO2_ao * 100
    CtO2_ao = alphaO2 * PO2_ao + CHb * SHbO2_ao * Hcrit
    CtO2_pa = alphaO2 * PO2_pa + CHb * SHbO2_pa * Hcrit
    CtO2_isf = alphaO2 * PO2_isf
    # if Vpc>Vpcmax:
    #     DL_O2 = ((0.397 mlSTPD/s/mmHg)+((0.0085 mlSTPD/s/mmHg**2)*PO2_ao)-((0.00013 mlSTPD/s/mmHg**3)*(PO2_ao**2)))+((5.1e-7 mlSTPD/s/mmHg**4)*(PO2_ao**3))
    # else:
    #     DL_O2 = (sqrt(Vpc/Vpcmax))*((0.397 mlSTPD/s/mmHg)+((0.0085 mlSTPD/s/mmHg**2)*PO2_ao)-((0.00013 mlSTPD/s/mmHg**3)*(PO2_ao**2)))+((5.1e-7 mlSTPD/s/mmHg**4)*(PO2_ao**3))
    if Vpc>Vpcmax:
        DL_O2 = ((0.397 * mlSTPD)+((0.0085 * mlSTPD**2)*PO2_ao)-((0.00013 * mlSTPD**3)*(PO2_ao**2)))+((5.1e-7 * mlSTPD**4)*(PO2_ao**3))
    else:
        DL_O2 = (np.sqrt(Vpc/Vpcmax))*((0.397  * mlSTPD)+((0.0085  * mlSTPD**2)*PO2_ao)-((0.00013  * mlSTPD**3)*(PO2_ao**2)))+((5.1e-7  * mlSTPD**4)*(PO2_ao**3))
    if PO2_isf<=0:
        V_O2 = 0
    else:
        V_O2 = (Vcytox * tweight * alphaO2 * PO2_isf) / (Kcytox + (alphaO2 * PO2_isf))
    O2flux = DL_O2 * (PA_O2 - PO2_ao) * 22400 # (22400 ml / mole)
    SHbCO2_ao = (PCO2_ao / P50_CO2) / ((PCO2_ao / P50_CO2) + 1)
    SHbCO2_pa = (PCO2_pa / P50_CO2) / ((PCO2_pa / P50_CO2) + 1)
    dSHbCO2dPCO2_ao = (PCO2_ao ** -1) * (PCO2_ao / P50_CO2) / ((PCO2_ao / P50_CO2) + 1) ** 2
    dSHbCO2dPCO2_pa = (PCO2_pa ** -1) * (PCO2_pa / P50_CO2) / ((PCO2_pa / P50_CO2) + 1) ** 2
    CtCO2_ao = alphaCO2 * PCO2_ao + CHb * SHbCO2_ao * Hcrit
    CtCO2_pa = alphaCO2 * PCO2_pa + CHb * SHbCO2_pa * Hcrit
    CtCO2_isf = alphaCO2 * PCO2_isf
    if Vpc>Vpcmax:
        DL_CO2 = 16.67 * mlSTPD # mlSTPD / s / mmHg
    else:
        DL_CO2 = (np.sqrt(Vpc/Vpcmax))*(16.67 * mlSTPD) # mlSTPD/s/mmHg)
    V_CO2 = RQ * V_O2
    CO2flux = DL_CO2 * (PA_CO2 - PCO2_ao) *22400 # (22400 ml / mole)
    Pao_O2 = (Pao - PH2O) * r_Pao_O2
    Pao_CO2 = (Pao - PH2O) * r_Pao_CO2
    Pao_N2 = (Pao - PH2O) - Pao_O2 - Pao_CO2
    Pmouth_CO2 = PD_CO2
    PD_N2 = (PDc + Pao) - PD_O2 - PD_CO2
    PC_N2 = (Pcc + Pao) - PC_O2 - PC_CO2
    PA_N2 = (PA + Pao) - PA_O2 - PA_CO2
    if Vpc>Vpcmax:
        DL_N2 = (0.25 * mlSTPD) # mlSTPD/s/mmHg)
    else:
        DL_N2 = (np.sqrt(Vpc/Vpcmax))*(0.25 * mlSTPD) # mlSTPD/s/mmHg)

    # airway mechanics
    Alvflux = (Pstp * Tbody / ((PA + 760) * Tstp)) * (O2flux + CO2flux)  # (760 mmHg)



    # ----------------- chemoreceptors ----------------- #
    # if PCO2_ao>=20:
    #     Kche = (0.35mmHg**(-1))*(PCO2_ao-20)
    # else:
    #     Kche = 0
    # fc = (Kche + ((1.4mmHg ** (-1)) * PCO2_ao * exp(-PO2_ao / Pche))) / 100
    if PCO2_ao>=20:
        Kche = (0.35)*(PCO2_ao-20)
    else:
        Kche = 0
    fc = (Kche + ((1.4) * PCO2_ao * np.exp(-PO2_ao / Pche))) / 100


    # blood gas handling
    # if (((Vcytox)/(Kcytox + CtO2_isf))>(51ml/min/g)):
    #     GO2isf = (51ml/min/g)
    # else:
    #     GO2isf = (Vcytox) / (Kcytox + CtO2_isf)
    if (((Vcytox)/(Kcytox + CtO2_isf))>(51)):
        GO2isf = (51)
    else:
        GO2isf = (Vcytox) / (Kcytox + CtO2_isf)
    DVRO11 = 1 + dSHbO2dPO2_ao * O2cap * Hbconc / alphaO2
    DVRO21 = 1 + dSHbO2dPO2_pa * O2cap * Hbconc / alphaO2
    pH_ao = -np.log(H_ao / HpNorm)
    pH_pa = -np.log(H_pa / HpNorm)
    pH_isf = -np.log(H_isf / HpNorm)



    # ----------------- pericardium ----------------- #
    # Ppcd = (K_pcd * exp((Vpcd - Vpcd_o) / phi_pcd)) - Px2 * (1 / (exp(Vpcd / Vx75) - 1))
    # Ppcdc = Ppcd + Pplc
    # Vpcd = Vrv + Vra + Vlv + Vla + perifl + Vmyo + Vcorcirc


    # ----------------- coronary circulation ----------------- #
    Pcorisfc = abs((Plvc - Ppcdc) / 2)
    Pcorao = Paop
    Pcorea = Vcorea / Ccorea - Px1 * (1 / (np.exp(Vcorea / Vx1) - 1))
    Pcorla = Vcorla / Ccorla - Px1 * (1 / (np.exp(Vcorla / Vx1) - 1))
    Pcorsa = Vcorsa / Ccorsa - Px1 * (1 / (np.exp(Vcorsa / Vx1) - 1))
    Pcorcap = Vcorcap / Ccorcap - Px1 * (1 / (np.exp(Vcorcap / Vx1) - 1))
    Pcorsv = Vcorsv / Ccorsv - Px1 * (1 / (np.exp(Vcorsv / Vx1) - 1))
    Pcorlv = Vcorlv / Ccorlv - Px1 * (1 / (np.exp(Vcorlv / Vx1) - 1))
    Pcorev = Vcorev / Ccorev - Px2 * (1 / (np.exp(Vcorev / Vx8) - 1))
    Pcoraoc = Paopc
    Pcoreac = Pcorea + Ppcdc
    Pcorlac = Pcorla + Pcorisfc
    Pcorsac = Pcorsa + Pcorisfc
    Pcorcapc = Pcorcap + Pcorisfc
    Pcorsvc = Pcorsv + Ppcdc
    Pcorlvc = Pcorlv + Ppcdc
    Pcorevc = Pcorev + Ppcdc
    Fcorao = (Pcoraoc - Pcoreac) / Rcorao
    Fcorea = (Pcoreac - Pcorlac) / Rcorea
    Fcorla = (Pcorlac - Pcorsac) / Rcorla
    Fcorsa = (Pcorsac - Pcorcapc) / Rcorsa
    Fcorcap = (Pcorcapc - Pcorsvc) / Rcorcap
    Fcorsv = (Pcorsvc - Pcorlvc) / Rcorsv
    Fcorlv = (Pcorlvc - Pcorevc) / Rcorlv
    Fcorev = (Pcorevc - Prac) / Rcorev
    Vcorcirc = Vcorao + Vcorea + Vcorla + Vcorsa + Vcorcap + Vcorsv + Vcorlv + Vcorev



    # ----------------- Differential Equations ----------------- #

    # heart
    Vla_dt = (Fpv - Fla)
    Vlv_dt = (Fla - Flv)
    Vra_dt = (Fvc - Fra + Fcorev)
    Vrv_dt = (Fra - Frv)
    COutput_dt = (Flv - COutput) / COtau

    # airway mechanics
    PplcFOL_dt = (Pplc - PplcFOL) / 0.001 # (0.001sec)
    # dV_dt = (Vcw - dV) / tau
    VA_dt = Qdotsm - Alvflux
    Vve_dt = VA_dt - Qdotve
    Vc_dt = Qdotco - Qdotsm
    Vcw_dt = VA_dt + Vc_dt

    # systemic
    PaodFOL_dt = (Paod - PaodFOL) / 0.0005 # (0.0005sec)
    MAP_dt = (Psa - MAP) / COtau
    Vaop_dt = (Paopc - (Vaop / Caop) + Px2 * (1 / (np.exp(Vaop / Vx8) - 1)) - PplcFOL) / Rtaop
    Paopc_dt = (Flv - Vaop_dt - Faop - Fcorao) * ((1 / Ccorao) + ((Px2 / (1)) * np.exp(Vcorao / Vx1) / (np.exp(Vcorao / Vx1) - 1) ** 2)) + PplcFOL_dt # (1ml)
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
    # Vpad_dt = Fpap - Fpad
    Vpa_dt = Fpad - Fps - Fpa
    Vpc_dt = Fpa - Fpc
    Vpv_dt = Fpc + Fps - Fpv
    Fpap_dt = (Ppapc - Ppadc - Fpap * Rpap) / Lpa
    Fpad_dt = (Ppadc - Ppac - Fpad * Rpad) / Lpad

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

    # baroreceptor
    Nbr_dt = Nbr_t
    # (a2 * a * (Nbr_t_dt)) + ((a2 + a) * Nbr_t) + Nbr = (K * (Paod) + (a1 * K * PaodFOL_dt))
    Nbr_t_dt = 1/(a2 * a) * (K * (Paod) + (a1 * K * PaodFOL_dt) - ((a2 + a) * Nbr_t) - Nbr)
    t_min = 0
    if t + t_min > L_hrv:
        N_hrv_dt = (-N_hrv + (K_hrv * Nbr(t - L_hrv))) / T_hrv
    else:
        N_hrv_dt = 0
    if t_min + t > L_hrs:
        N_hrs_dt = (-N_hrs + (K_hrs * Nbr(t - L_hrs))) / T_hrs
    else:
        N_hrs_dt = 0
    if t_min + t > L_con:
        N_con_dt = (-N_con + (K_con * Nbr(t - L_con))) / T_con
    else:
        N_con_dt = 0
    if t_min + t > L_vaso:
        N_vaso_dt = (-N_vaso + (K_vaso * Nbr(t - L_vaso))) / T_vaso
    else:
        N_vaso_dt = 0

    # total fluid
    Vtot = Vra + Vrv + Vpap + Vpad + Vpa + Vpc + Vpv + Vla + Vlv + Vaop + Vaod +Vsa + Vsap + Vsc + Vsv + Vvc + perifl + Vcorcirc
    VBcirc = Vtot - perifl
    SysArtVol = Vaop + Vaod + Vsa + Vsap
    SysVenVol = Vsv + Vvc
    PulArtVol = Vpap + Vpad + Vpa
    PulVenVol = Vpv

    dy = np.array([
        Vla_dt,
        Vlv_dt,
        Vra_dt,
        Vrv_dt,
        COutput_dt,
        Paopc_dt,
        MAP_dt,
        Vaop_dt,
        Vaod_dt,
        Vsap_dt,
        Vsa_dt,
        Vsc_dt,
        Vsv_dt,
        Vvc_dt,
        Faop_dt,
        Faod_dt,
        Vpap_dt,
        Vpad_dt,
        Vpa_dt,
        Vpc_dt,
        Vpv_dt,
        Fpap_dt,
        Fpad_dt,
        dV_dt,
        PaodFOL_dt,
        PplcFOL_dt,
        Vcw_dt,
        VA_dt,
        Vve_dt,
        Vc_dt,
        PO2_ao_dt,
        PO2_pa_dt,
        PO2_isf_dt,
        PCO2_ao_dt,
        PCO2_pa_dt,
        PCO2_isf_dt,
        PD_O2_dt,
        PC_O2_dt,
        PA_O2_dt,
        PD_CO2_dt,
        PC_CO2_dt,
        PA_CO2_dt,
        gc_dt,
        PBC_pc_dt,
        PBC_sc_dt,
        HCO3_ao_dt,
        HCO3_pa_dt,
        HCO3_isf_dt,
        Hout_dt,
        H_ao_dt,
        H_pa_dt,
        H_isf_dt,
        Vcorao_dt,
        Vcorea_dt,
        Vcorla_dt,
        Vcorsa_dt,
        Vcorcap_dt,
        Vcorsv_dt,
        Vcorlv_dt,
        Vcorev_dt,
        Nbr_dt,
        Nbr_t_dt,
        N_hrv_dt,
        N_hrs_dt,
        N_con_dt,
        N_vaso_dt,
    ])

    return dy


def call_cardio(args, params, debug=False):

    # load input data
    mlSTPD = 1 / 22400
    dt = 0.01
    t_max = 1
    t_meas = np.arange(0, t_max, dt)

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
    tshift = t_meas[0]
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
    tresp = t_meas[0]
    A = params.loc["KA"] * gc + params.loc["DA"]
    B = params.loc["KB"] * gc + params.loc["DB"]
    C = params.loc["KC"] * gc + params.loc["DC"]
    if C / (2 * np.pi) > (120):
        RespRcont = 120
    else:
        RespRcont = C / (2 * np.pi)
    RespR = RespRcont
    Pcw = (A * np.sin(2 * np.pi * RespR * (t_meas[0] - tresp)) - B)
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
        Vla,
        Vlv,
        Vra,
        Vrv,
        COutput,
        Paopc,
        MAP,
        Vaop,
        Vaod,
        Vsap,
        Vsa,
        Vsc,
        Vsv,
        Vvc,
        Faop,
        Faod,
        Vpap,
        Vpad,
        Vpa,
        Vpc,
        Vpv,
        Fpap,
        Fpad,
        dV,
        PaodFOL,
        PplcFOL,
        Vcw,
        VA,
        Vve,
        Vc,
        PO2_ao,
        PO2_pa,
        PO2_isf,
        PCO2_ao,
        PCO2_pa,
        PCO2_isf,
        PD_O2,
        PC_O2,
        PA_O2,
        PD_CO2,
        PC_CO2,
        PA_CO2,
        gc,
        PBC_pc,
        PBC_sc,
        HCO3_ao,
        HCO3_pa,
        HCO3_isf,
        Hout,
        H_ao,
        H_pa,
        H_isf,
        Vcorao,
        Vcorea,
        Vcorla,
        Vcorsa,
        Vcorcap,
        Vcorsv,
        Vcorlv,
        Vcorev,
        Nbr,
        Nbr_t,
        N_hrv,
        N_hrs,
        N_con,
        N_vaso,
    ])

    Nbr_list = []
    Nbr_list_idx = []

    t_list = []
    Pra_list, Prv_list, Pla_list, Plv_list = [], [], [], []
    Vra_list, Vrv_list, Vla_list, Vlv_list = [], [], [], []

    F_con = params.loc["a_con"] + (params.loc["b_con"] / (np.exp(params.loc["tau_con"] * (N_con - params.loc["No_con"])) + 1.0))
    afs_con = params.loc["amin"] + (params.loc["Ka"] * F_con)
    F_hrs = params.loc["a_hrs"] + (params.loc["b_hrs"] / (np.exp(params.loc["tau_hrs"] * (N_hrs - params.loc["No_hrs"])) + 1.0))
    F_hrv = params.loc["a_hrv"] + (params.loc["b_hrv"] / (np.exp(params.loc["tau_hrv"] * (N_hrv - params.loc["No_hrv"])) + 1.0))
    HRcont = (params.loc["h1"] + (params.loc["h2"]*F_hrs)-(params.loc["h3"]*F_hrs**2)-(params.loc["h4"]*F_hrv)+ (params.loc["h5"]*F_hrv**2)-(params.loc["h6"]*F_hrv*F_hrs))

    trelv = t_meas[0] - tshift - params.loc["PRint"]
    if 0.0 <= trelv and trelv < Tsv:
        ylv = (1.0 - np.cos(np.pi*trelv/Tsv))/2.0
    elif Tsv<=trelv and trelv<1.5*Tsv:
        ylv = (1.0 + np.cos(2.0*np.pi*(trelv-Tsv)/Tsv))/2.0
    else:
        ylv = 0
    yrv = ylv
    Vlvr = (1 - ylv) * (params.loc["Vlvrd"] - params.loc["Vlvrs"]) + params.loc["Vlvrs"]
    Elv= ((params.loc["Emaxlv"]-params.loc["Eminlv"])*ylv) + params.loc["Eminlv"]
    Plv = afs_con2 * (Vlv - Vlvr) * Elv - params.loc["Px2"] * (1 / (np.exp((Vlv) / params.loc["Vx8"]) - 1))
    Vcorcirc = Vcorao + Vcorea + Vcorla + Vcorsa + Vcorcap + Vcorsv + Vcorlv + Vcorev
    Vpcd = Vrv + Vra + Vlv + Vla + params.loc["perifl"] + params.loc["Vmyo"] + Vcorcirc
    Ppcd = (params.loc["K_pcd"] * np.exp((Vpcd - params.loc["Vpcd_o"]) / params.loc["phi_pcd"])) - params.loc["Px2"] * (1 / (np.exp(Vpcd / params.loc["Vx75"]) - 1))
    Ppl = Pcw
    Pplc = Ppl
    Ppcdc = Ppcd + Pplc
    Plvc = Plv + Ppcdc
    Erv = ((params.loc["Emaxrv"] - params.loc["Eminrv"]) * yrv) + params.loc["Eminrv"]
    Prv = afs_con2 * (Vrv - params.loc["Vrvr"]) * Erv - params.loc["Px2"] * (1 / (np.exp(Vrv / params.loc["Vx8"]) - 1))
    Prvc = Prv + Ppcdc

    params_list = [[tshift], [HR], [HRcont], [afs_con],
                   [ylv], [Plvc], [Prvc], [Vlv], [Vrv],
                   [Tsa], [Tsv], [afs_con2], [RespR], [tresp]]

    ODE_args = (
        t_meas, dt, mlSTPD,
        params_list,

        params.loc["Vlvrd"],
        params.loc["Vlvrs"],
        params.loc["Vrvrd"],
        params.loc["Vrvrs"],
        params.loc["Vlard"],
        params.loc["Vlars"],
        params.loc["Vrard"],
        params.loc["Vrars"],
        params.loc["Rra"],
        params.loc["Rla"],
        params.loc["PRint"],
        params.loc["Emaxlv"],
        params.loc["Eminlv"],
        params.loc["Emaxrv"],
        params.loc["Eminrv"],
        params.loc["Emaxra"],
        params.loc["Eminra"],
        params.loc["Emaxla"],
        params.loc["Eminla"],
        params.loc["Pbs"],
        params.loc["Vmyo"],
        params.loc["Punit"],
        params.loc["TsvK"],
        params.loc["TsaK"],
        params.loc["Rav"],
        params.loc["Raop"],
        params.loc["Rcrb"],
        params.loc["Raod"],
        params.loc["Rtaop"],
        params.loc["Rtaod"],
        params.loc["Rsap"],
        params.loc["Rsc"],
        params.loc["Rsv"],
        params.loc["Caop"],
        params.loc["Caod"],
        params.loc["Csap"],
        params.loc["Csc"],
        params.loc["Laop"],
        params.loc["Laod"],
        params.loc["Kc"],
        params.loc["Do"],
        params.loc["Vsa_o"],
        params.loc["Vsa_max"],
        params.loc["Kp1"],
        params.loc["Kp2"],
        params.loc["Kr"],
        params.loc["Rsao"],
        params.loc["tau_p"],
        params.loc["Kv"],
        params.loc["Vmax_sv"],
        params.loc["D2"],
        params.loc["K1"],
        params.loc["K2"],
        params.loc["KR"],
        params.loc["Ro"],
        params.loc["Vo"],
        params.loc["Vmax_vc"],
        params.loc["Vmin_vc"],
        params.loc["COtau"],
        params.loc["Px2"],
        params.loc["Vx8"],
        params.loc["Vx75"],
        params.loc["Vx1"],
        params.loc["Px1"],
        params.loc["Rpuv"],
        params.loc["Rtpap"],
        params.loc["Rtpad"],
        params.loc["Rpap"],
        params.loc["Rpad"],
        params.loc["Rps"],
        params.loc["Rpa"],
        params.loc["Rpc"],
        params.loc["Rpv"],
        params.loc["Ctpap"],
        params.loc["Ctpad"],
        params.loc["Cpa"],
        params.loc["Cpc"],
        params.loc["Cpv"],
        params.loc["Lpa"],
        params.loc["Lpad"],
        params.loc["a"],
        params.loc["a1"],
        params.loc["a2"],
        params.loc["K"],
        params.loc["K_hrv"],
        params.loc["T_hrv"],
        params.loc["L_hrv"],
        params.loc["a_hrv"],
        params.loc["tau_hrv"],
        params.loc["No_hrv"],
        params.loc["K_hrs"],
        params.loc["T_hrs"],
        params.loc["L_hrs"],
        params.loc["a_hrs"],
        params.loc["tau_hrs"],
        params.loc["No_hrs"],
        params.loc["K_con"],
        params.loc["T_con"],
        params.loc["L_con"],
        params.loc["a_con"],
        params.loc["b_con"],
        params.loc["tau_con"],
        params.loc["No_con"],
        params.loc["K_vaso"],
        params.loc["T_vaso"],
        params.loc["L_vaso"],
        params.loc["a_vaso"],
        params.loc["tau_vaso"],
        params.loc["No_vaso"],
        params.loc["h1"],
        params.loc["h2"],
        params.loc["h3"],
        params.loc["h4"],
        params.loc["h5"],
        params.loc["h6"],
        params.loc["amin"],
        params.loc["bmin"],
        params.loc["Ka"],
        params.loc["Kb"],
        params.loc["RV"],
        params.loc["VD"],
        params.loc["Ac"],
        params.loc["Al"],
        params.loc["As"],
        params.loc["Au"],
        params.loc["Bc"],
        params.loc["Bcp"],
        params.loc["Bl"],
        params.loc["Bs"],
        params.loc["Cve"],
        params.loc["Kc_air"],
        params.loc["Kl"],
        params.loc["Ks"],
        params.loc["Ku"],
        params.loc["Rve"],
        params.loc["Vstar"],
        params.loc["Vcmax"],
        params.loc["tau"],
        params.loc["KA"],
        params.loc["KB"],
        params.loc["KC"],
        params.loc["DA"],
        params.loc["DB"],
        params.loc["DC"],
        params.loc["Ccw"],
        params.loc["Tbody"],
        params.loc["Pstp"],
        params.loc["Tstp"],
        params.loc["nH"],
        params.loc["P50_O2"],
        params.loc["CHb"],
        params.loc["Hcrit"],
        params.loc["alphaO2"],
        params.loc["Visf"],
        params.loc["PS"],
        params.loc["Vcytox"],
        params.loc["Kcytox"],
        params.loc["P50_CO2"],
        params.loc["alphaCO2"],
        params.loc["RQ"],
        params.loc["PH2O"],
        params.loc["Vpcmax"],
        params.loc["Pao"],
        params.loc["r_Pao_O2"],
        params.loc["r_Pao_CO2"],
        params.loc["tauv"],
        params.loc["Pche"],
        params.loc["kp1"],
        params.loc["km1"],
        params.loc["K1bgh"],
        params.loc["K2bgh"],
        params.loc["K3bgh"],
        params.loc["K5bgh"],
        params.loc["K6bgh"],
        params.loc["kp5"],
        params.loc["CFb"],
        params.loc["CFt"],
        params.loc["Betab"],
        params.loc["Betat"],
        params.loc["O2cap"],
        params.loc["Hbconc"],
        params.loc["Cheme"],
        params.loc["HpNorm"],
        params.loc["facid"],
        params.loc["tweight"],
        params.loc["tauHout"],
        params.loc["PS_H"],
        params.loc["PS_HCO3"],
        params.loc["K_pcd"],
        params.loc["phi_pcd"],
        params.loc["Vpcd_o"],
        params.loc["perifl"],
        params.loc["Rcorao"],
        params.loc["Rcorea"],
        params.loc["Rcorla"],
        params.loc["Rcorsa"],
        params.loc["Rcorcap"],
        params.loc["Rcorsv"],
        params.loc["Rcorlv"],
        params.loc["Rcorev"],
        params.loc["Ccorao"],
        params.loc["Ccorea"],
        params.loc["Ccorla"],
        params.loc["Ccorsa"],
        params.loc["Ccorcap"],
        params.loc["Ccorsv"],
        params.loc["Ccorlv"],
        params.loc["Ccorev"],
    )

    max_time_step = 10
    sol = solve_ivp(fun=ODE,
                    t_span=t_meas,
                    y0=y0,
                    args=ODE_args,
                    t_eval=t_meas,
                    # first_step=0.02,
                    # rtol=1e-1, atol=1e-2,
                    method="LSODA")
    return


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
