import os

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

from ._baroreceptor import _f, _b_vaso, _firing_frequency, _n_change
from ._blood import _total_blood_volume, _heart_volume, _coronary_volume, _systemic_arterial_volume, \
    _systemic_venous_volume, _pulmonary_arterial_volume
from ._coronary import _p_eq, _pcor, _pcorc, _fcor, _vcor
from ._heart import _trigger_A, _trigger_B, t_rel, _yi, _ei, _v0, _pa, _fi, _volume_change, _pv, _emaxv
from ._pulmonary import _ppap, _ppad, _pp, _fp, _vp, _fpa
from ._systemic import _cardiac_output, _stroke_volume, _arterial_blood_pressure, _kv, _map, _aortic_afterload, _psap, \
    _psa_a, _psa_p, _psa, _psc, _psv, _pvc, _forward_flow, _abp_change, _aortic_flow_change, _vaop_change, _v_change, \
    _pulmonary_valve_flow_change, _pressure_aop_change, _rsa, _rvc, _mapmeas_change, _comea_change


def ODE(t, y,

        x, tHB, HP, tmeas, ABPmeas, PAFmeas, Nbr_list, Nbr_list_idx,

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
    
    to_L_min = 10 ** (-3) * 60
    to_ml_sec = 10 ** 3 / 60
    to_min_L = 60 / 10 ** 3
    to_ml = 10 ** 3
    to_min = 60
    to_min2_L = 10 ** 3 / 60 ** 2

    HRa, HRv, Vra, Vrv, Vla, Vlv, Tsv, Tsa, Vvarlvs0, Vvarrvs0, n, m, tRwave, tPwave,\
    MAPmeas, Faop, Faod, Frv_sm, Vaop, Vaod, Vsa, Vsap, Vsc, Vsv, Vvc, Paop, AOFmod, ABPfol, COmea,\
    Vpap, Vpad, Vpa, Vpc, Vpv, Fpap, Fpad,\
    Vcorepi, Vcorintra, Vcorcap, Vcorvn,\
    Nbr, Nbr_t, N_con, N_vaso, af_con2 = y

    n = int(n)
    m = int(m)

    f_con = a_con + b_con / (np.exp(tau_con * (N_con - No_con)) + 1.0)
    af_con = amin + (Ka * f_con)

    # heart
    resultA = _trigger_A(t, tHB[n+1], HP[n+1], PRint, offv, Ts1a, Ts2)
    if resultA is not None:
        HRa, Tsa, tPwave = resultA
        n = n + 1

    resultB = _trigger_B(t, tHB[m+1], HP[m+1], offv, Ts1v, Ts2, n,
                         Vlv, Vlvd0, EDVLV, Vlvs0,
                         Vrv, Vrvd0, EDVRV, Vrvs0,
                         af_con)
    if resultB is not None:
        HRv, Tsv, tRwave, Vvarlvs0, Vvarrvs0, af_con2, m = resultB

    HR = HRv

    Emaxlv = _emaxv(KElv, Emaxlv1)
    Emaxrv = _emaxv(KErv, Emaxrv1)

    # ----------- heart ----------- #
    ta_rel = t_rel(t, tPwave)
    tv_rel = t_rel(t, tRwave)
    # activation functions
    ya = _yi(ta_rel, Tsa)
    yv = _yi(tv_rel, Tsv)
    # elastances
    Era = _ei(Emaxra, Eminra, ya)
    Erv = _ei(Emaxrv, Eminrv, yv)
    Ela = _ei(Emaxla, Eminla, ya)
    Elv = _ei(Emaxlv, Eminlv, yv)
    # unstressed volumes
    Vra0 = _v0(ya, Vrad0, Vras0)
    Vrv0 = _v0(yv, Vrvd0, Vvarrvs0)
    Vla0 = _v0(ya, Vlad0, Vlas0)
    Vlv0 = _v0(yv, Vlvd0, Vvarlvs0)
    # pressures
    Pra = _pa(Era, Vra, Vra0, Kxp, Kxv)
    Prv = _pv(Erv, Vrv, Vrv0, af_con2, Kxp, Kxv)
    Pla = _pa(Ela, Vla, Vla0, Kxp, Kxv)
    Plv = _pv(Elv, Vlv, Vlv0, af_con2, Kxp, Kxv) # TODO: tbc

    Ppap = _ppap(Prv, Rtpap, Rrv, 
                 Fpap * to_ml_sec, 
                 Vpap, Vpap0, Cpap, Kxp, Kxv)
    # flows [L/min]
    Fra = _fi(Pra, Prv, Rra) * to_L_min
    Frv = _fi(Prv, Ppap, Rrv) * to_L_min
    Fla = _fi(Pla, Plv, Rla) * to_L_min
    Flv = _fi(Plv, Paop, Rlv) * to_L_min

    # ----------- systemic circulation ----------- #
    Pvc = _pvc(Vvc, Vvc0, Vmin_vc, K1, K2, D2, Kxp, Kxv)
    # misc
    # [L/min]
    COmod = _cardiac_output(Frv_sm)
    # [ml]
    SV = _stroke_volume(COmod, HR) * to_ml

    # TODO: check this arrangement
    i = np.argmin(np.abs(tmeas - (t + offv)))
    ABPshift = ABPmeas[i]

    Kv = _kv(Kv1, Ksv)
    MAPmod = _map(Rtaod, Rcrb,
                  AOFmod * to_ml_sec,
                  Faod * to_ml_sec,
                  Vaod, Vaod0, Caod, Kxp, Kxv, Pvc)

    b_vaso = _b_vaso(a_vaso)
    f_vaso = _f(a_vaso, b_vaso, tau_vaso, N_vaso, No_vaso)
    # pressures
    Paod = _aortic_afterload(ABPshift)
    Psap = _psap(Vsap, Vsap0, Csap, Kxp, Kxv)
    Psa_a = _psa_a(Kc, Vsa, Vsa0, Do)
    Psa_p = _psa_p(Kp1, Kp2, tau_p, Vsa, Vsa0)
    Psa = _psa(f_vaso, Psa_a, Psa_p)
    Psc = _psc(Vsc, Vsc0, Csc, Kxp, Kxv)
    Psv = _psv(Kv, Vmax_sv, Vsv)
    # Pvc = _pvc(Vvc, Vvc0, Vmin_vc, K1, K2, D2, Kxp, Kxv)

    Rsa = _rsa(Kr, f_vaso, Vsa, Vsa_max, Rsa0)
    Rvc = _rvc(KR, Vmax_vc, Vvc, R0)

    # flows [L/min]
    Fcrb = _forward_flow(MAPmod, Pvc, Rcrb) * to_L_min
    Fsap = _forward_flow(Psap, Psa, Rsap) * to_L_min
    Fsa = _forward_flow(Psa, Psc, Rsa) * to_L_min
    Fsc = _forward_flow(Psc, Psv, Rsc) * to_L_min
    Fsv = _forward_flow(Psv, Pvc, Rsv) * to_L_min
    Fvc = _forward_flow(Pvc, Pra, Rvc) * to_L_min

    Pcorepi = _p_eq(Paop)
    Pcorintra = _pcor(Vcorintra, Vcorintra0, Ccorintra, Kxp1, Kxv1)
    Pcorintrac = _pcorc(Pcorintra, Plv)
    # [L/min]
    Fcorepi = _fcor(Pcorepi, Pcorintrac, Rcorepi) * to_L_min
    # differential equations
    d_ABPfol_dt = _abp_change(ABPshift, ABPfol, tauABP)
    # [L/min^2]
    d_AOFmod_dt = _aortic_flow_change(MAPmeas, MAPmod, KCOMAP)
    # [ml/sec]
    d_Vaop_dt = _vaop_change(Paop, Vaop, Vaop0, Caop, Rtaop)
    d_Vaod_dt = _v_change(AOFmod, Faod + Fcrb) * to_ml_sec
    d_Vsa_dt = _v_change(Fsap, Fsa) * to_ml_sec
    d_Vsap_dt = _v_change(Faod, Fsap) * to_ml_sec
    d_Vsc_dt = _v_change(Fsa, Fsc) * to_ml_sec
    d_Vsv_dt = _v_change(Fsc, Fsv) * to_ml_sec
    d_Vvc_dt = _v_change(Fsv + Fcrb, Fvc) * to_ml_sec
    # [L/min^2]
    d_Frv_sm_dt = _pulmonary_valve_flow_change(Frv, Frv_sm,
                                               tauCO * to_min)
    d_Faop_dt = _forward_flow(Paop,
                              Faop * Raop * to_min_L + Paod,
                              Laop * to_min2_L)
    d_Faod_dt = _forward_flow(MAPmod,
                              Faod * Raod * to_min_L - Psap,
                              Laod * to_min2_L)
    # [mmHg/sec]
    d_Paop_dt = _pressure_aop_change(Flv * to_ml_sec, # [L/min]
                                     d_Vaop_dt, # [ml/sec]
                                     Faop * to_ml_sec, # [L/min]
                                     Fcorepi * to_ml_sec, # [L/min]
                                     Ccorepi) # ml*mmHg^(-1)

    # ----------- pulmonary circulation ----------- #

    # pressures
    # [mmHg]
    Ppad = _ppad(Vpad, Vpad0, Kxp, Kxv,
                 Fpap * to_ml_sec, # [L/min]
                 Rtpad,
                 Fpad * to_ml_sec, # [L/min]
                 Cpad)
    Ppa = _pp(Vpa, Vpa0, Cpa, Kxp, Kxv)
    Ppc = _pp(Vpc, Vpc0, Cpc, Kxp, Kxv)
    Ppv = _pp(Vpv, Vpv0, Cpv, Kxp, Kxv)

    # flows
    Fps = _fp(Ppa, Ppv, Rps) * to_L_min
    Fpa = _fp(Ppa, Ppc, Rpa) * to_L_min
    Fpc = _fp(Ppc, Ppv, Rpc) * to_L_min
    Fpv = _fp(Ppv, Pla, Rpv) * to_L_min

    # differential equations
    # [ml/sec]
    d_Vpad_dt = _vp(Fpap, Fpad) * to_ml_sec
    d_Vpap_dt = _vp(Frv, Fpap) * to_ml_sec
    d_Vpa_dt = _vp(Fpad, Fps + Fpa) * to_ml_sec
    d_Vpc_dt = _vp(Fpa, Fpc) * to_ml_sec
    d_Vpv_dt = _vp(Fpc + Fps, Fpv) * to_ml_sec
    # [L/min^2]
    d_Fpap_dt = _fpa(Ppap,
                     Ppad + Fpap * Rpap * to_min_L,
                     Lpap * to_min2_L)
    d_Fpad_dt = _fpa(Ppad,
                     Ppa + Fpad * Rpad * to_min_L,
                     Lpad)

    # ----------- coronary circulation ----------- #
    # pressures
    # Pcorepi = _p_eq(Paop)
    # Pcorintra = _pcor(Vcorintra, Vcorintra0, Ccorintra, Kxp1, Kxv1)
    # Pcorintrac = _pcorc(Pcorintra, Plv)
    Pcorcap = _pcor(Vcorcap, Vcorcap0, Ccorcap, Kxp1, Kxv1)
    Pcorvn = _pcor(Vcorvn, Vcorvn0, Ccorvn, Kxp, Kxv)
    Pcorcapc = _pcorc(Pcorcap, Plv)
    Pcorvnc = _p_eq(Pcorvn)

    # flows
    # Fcorepi = _fcor(Pcorepi, Pcorintrac, Rcorepi)
    # [L/min]
    Fcorintra = _fcor(Pcorintrac, Pcorcapc, Rcorintra) * to_L_min
    Fcorcap = _fcor(Pcorcapc, Pcorvnc, Rcorcap) * to_L_min
    Fcorvn = _fcor(Pcorvnc, Pra, Rcorvn) * to_L_min

    # differential equations
    # [ml/sec]
    d_Vcorepi_dt = _vcor(Flv, # [L/min]
                         d_Vaop_dt * to_L_min + Faop + Fcorepi) * to_ml_sec # [ml/sec] * [L/min] * [L/min]
    d_Vcorintra_dt = _vcor(Fcorepi, Fcorintra) * to_ml_sec
    d_Vcorcap_dt = _vcor(Fcorintra, Fcorcap) * to_ml_sec
    d_Vcorvn_dt = _vcor(Fcorcap, Fcorvn) * to_ml_sec

    # ----------- baroreceptor ----------- #
    # Equation 2 order!

    # ----------- blood ----------- #
    Vcorcirc = _coronary_volume(Vcorepi, Vcorintra, Vcorcap, Vcorvn)
    Vheart = _heart_volume(Vra, Vrv, Vla, Vlv, Vcorcirc)
    VSysArt = _systemic_arterial_volume(Vaop, Vaod, Vsap, Vsa)
    VSysVen = _systemic_venous_volume(Vsv, Vvc)
    VPulArt = _pulmonary_arterial_volume(Vpap, Vpad, Vpa)
    TBV = _total_blood_volume(Vheart, VPulArt, Vpc, Vpv, VSysArt, Vsc, VSysVen)

    # heart
    # [ml/sec]
    d_Vra_dt = _volume_change(Fvc + Fcorvn, Fra) * to_ml_sec
    d_Vrv_dt = _volume_change(Fra, Frv) * to_ml_sec
    d_Vla_dt = _volume_change(Fpv, Fla) * to_ml_sec
    d_Vlv_dt = _volume_change(Fla, Flv) * to_ml_sec

    # [mmHg/sec]
    d_MAPmeas_dt = _mapmeas_change(ABPshift, MAPmeas, tauMAP)
    # [L/min^2]
    d_COmea_dt = _comea_change(PAFmeas[n], COmea, tauCO * to_min)
    # [mmHg/sec]
    d_ABPfol_dt = _abp_change(ABPshift, ABPfol, tauABP)

    # [sec^-2]
    d_Nbr_dt = Nbr_t
    # [sec^-3]
    d_Nbr_t_dt = _firing_frequency(d_ABPfol_dt, Nbr_t, Nbr, ABPshift, a, a1, a2, K)
    # [sec^-2]
    d_N_con_dt = _n_change(t, tHB[1], l_con, N_con, K_con, Nbr_list, Nbr_list_idx, T_con)
    d_N_vaso_dt = _n_change(t, tHB[1], l_vaso, N_vaso, K_vaso, Nbr_list, Nbr_list_idx, T_vaso)

    dy = np.array([
        HRa, HRv,
        d_Vra_dt, d_Vrv_dt, d_Vla_dt, d_Vlv_dt, Tsv, Tsa, Vvarlvs0, Vvarrvs0, n, m, tRwave, tPwave,
        d_MAPmeas_dt, d_Faop_dt, d_Faod_dt, d_Frv_sm_dt, d_Vaop_dt, d_Vaod_dt, d_Vsa_dt, d_Vsap_dt, d_Vsc_dt, d_Vsv_dt, d_Vvc_dt, d_Paop_dt, d_AOFmod_dt, d_ABPfol_dt, d_COmea_dt,
        d_Vpap_dt, d_Vpad_dt, d_Vpa_dt, d_Vpc_dt, d_Vpv_dt, d_Fpap_dt, d_Fpad_dt,
        d_Vcorepi_dt, d_Vcorintra_dt, d_Vcorcap_dt, d_Vcorvn_dt,
        d_Nbr_dt, d_Nbr_t_dt, d_N_con_dt, d_N_vaso_dt, af_con2
    ])
    types_dict = {i: type(dy[i]) for i, _ in enumerate(dy)}

    Nbr_list.append(Nbr)
    Nbr_list_idx.append(t)

    print(f"t={t:.2f}: ya[{ya:.2f}] ta_rel[{ta_rel:.2f}] tPwave[{tPwave:.2f}]")
    print(f"t={t:.2f}: Ela[{Ela:.2f}] Vla[{Vla:.2f}]")
    print(f"t={t:.2f}: Ppv[{Ppv:.2f}] Pla[{Pla:.2f}]")
    print(f"t={t:.2f}: Fpv[{Fpv:.2f}] Fla[{Fla:.2f}] Flv[{Flv:.2f}]")
    print(f"t={t:.2f}: Vra[{Vra:.2f}] Vrv[{Vrv:.2f}] Vla[{Vla:.2f}] Vlv[{Vlv:.2f}] TBV[{TBV:.2f}]\n")

    if t > 1.02:
        print()

    return dy


def call_cardio(args, params):

    abp_df = pd.read_csv("cardiacdata2.csv")
    hp_df = pd.read_csv("cardiacdatab.csv")

    x = hp_df["x"].values
    tHB = hp_df["t"].values
    HP = hp_df["HP"].values

    # t1 = np.arange(tHB[0], tHB[-1]-1, 0.02)
    tmeas = abp_df["x"].values
    ABPmeas = abp_df["ABP"].values

    ABPshift = _arterial_blood_pressure(ABPmeas[0], tmeas[0], params.loc["offv", "value"])

    # TODO: we need it!
    PAFmeas = ABPmeas

    ########## initial values
    # heart
    HRa = 1 / HP[np.min(x)]
    HRv = 1 / HP[np.min(x)]
    Vra = 77.2392888
    Vrv = 177.14423541
    Vla = 60.99697623
    Vlv = 126.53255174
    Tsv = params.loc["Ts1v", "value"] * np.sqrt(1.0 / HRv / 1.0)
    Tsa = params.loc["Ts1a", "value"] * np.sqrt(1.0 / HRa / 1.0)
    Vvarlvs0 = params.loc["Vlvs0", "value"]
    Vvarrvs0 = params.loc["Vrvs0", "value"]
    n = np.min(x)
    m = np.min(x)
    tRwave = tHB[1] - params.loc["offv", "value"]
    tPwave = tHB[1] - params.loc["PRint", "value"] - params.loc["offv", "value"]

    # systemic circulation
    MAPmeas = ABPshift
    Faop = -0.27112497
    Faod = 5.46990935
    Frv_sm = 6.42512266
    Vaop = 30.20137958
    Vaod = 83.39476893
    Vsa = 522.12606521
    Vsap = 191.09054537
    Vsc = 256.06464838
    Vsv = 2942.0057
    Vvc = 244.21
    Paop = 82.94244954
    AOFmod = 6.25466118
    ABPfol = ABPshift
    COmea = Frv_sm

    # pulmonary circulation
    Vpap = 30.43790817
    Vpad = 59.76546862
    Vpa = 51.49758158
    Vpc = 104.87877646
    Vpv = 307.91244142
    Fpap = 1.22033388
    Fpad = 2.38478489


    Pcorepi = Paop
    # coronary circulation
    Vcorepi = params.loc["Ccorepi", "value"] * Pcorepi + params.loc["Vcorepi0", "value"]
    Vcorintra = 9.91970364
    Vcorcap = 10.0892486
    Vcorvn = 23.86304853

    # baroreceptor
    Nbr = 82.23159612
    Nbr_t = 0.00133763
    N_con = 93.34406906
    N_vaso = 93.22417668
    f_con = params.loc["a_con", "value"] + (params.loc["b_con", "value"] / (np.exp(params.loc["tau_con", "value"] * (N_con - params.loc["No_con", "value"])) + 1.0))
    af_con = params.loc["amin", "value"] + (params.loc["Ka", "value"] * f_con)
    af_con2 = af_con
    #########

    Vcorcirc = _coronary_volume(Vcorepi, Vcorintra, Vcorcap, Vcorvn)
    Vheart = _heart_volume(Vra, Vrv, Vla, Vlv, Vcorcirc)
    VSysArt = _systemic_arterial_volume(Vaop, Vaod, Vsap, Vsa)
    VSysVen = _systemic_venous_volume(Vsv, Vvc)
    VPulArt = _pulmonary_arterial_volume(Vpap, Vpad, Vpa)
    TBV = _total_blood_volume(Vheart, VPulArt, Vpc, Vpv, VSysArt, Vsc, VSysVen)

    # initial condition for the ODE solver
    y0 = np.array([
        HRa, HRv, Vra, Vrv, Vla, Vlv, Tsv, Tsa, Vvarlvs0, Vvarrvs0, n, m, tRwave, tPwave,
        MAPmeas, Faop, Faod, Frv_sm, Vaop, Vaod, Vsa, Vsap, Vsc, Vsv, Vvc, Paop, AOFmod, ABPfol, COmea,
        Vpap, Vpad, Vpa, Vpc, Vpv, Fpap, Fpad,
        Vcorepi, Vcorintra, Vcorcap, Vcorvn,
        Nbr, Nbr_t, N_con, N_vaso, af_con2,
    ])

    Nbr_list = []
    Nbr_list_idx = []

    ODE_args = (
        x, tHB, HP, tmeas, ABPmeas, PAFmeas, Nbr_list, Nbr_list_idx,

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

    # i = np.argmin(np.abs(tmeas - (tmeas[5] + params.loc["offv", "value"])))
    # ABPmeas[]

    # fig =

    sol = solve_ivp(fun=ODE, t_span=[tHB[1], np.max(tmeas)], y0=y0,
                    args=ODE_args,
                    # first_step=0.02, #t_eval=tmeas,
                    # rtol=1e-1, atol=1e-2,
                    method="RK23")

    return
