import os

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

from ._blood import _total_blood_volume, _heart_volume, _coronary_volume, _systemic_arterial_volume, \
    _systemic_venous_volume, _pulmonary_arterial_volume
from ._coronary import _p_eq, _pcor, _pcorc, _fcor, _vcor
from ._heart import _trigger_A, _trigger_B, t_rel, _yi, _ei, _v0, _pa, _fi, _volume_change, _pv
from ._pulmonary import _ppap, _ppad, _pp, _fp, _vp, _fpa
from ._systemic import _cardiac_output, _stroke_volume, _arterial_blood_pressure, _kv, _map, _aortic_afterload, _psap, \
    _psa_a, _psa_p, _psa, _psc, _psv, _pvc, _forward_flow, _abp_change, _aortic_flow_change, _vaop_change, _v_change, \
    _pulmonary_valve_flow_change, _pressure_aop_change


def ODE(t, y,

        x, tHB, HP, tmeas, ABPmeas,

        # parameters
        tauC0, Emaxlv, Vvarlvs0, Vvarrvs0,

        # heart
        # ta_rel, tv_rel, Tsv, Tsa, n, m, HR, HRa, HRv, tRwave, tPwave, Vvarlvs0, Vvarrvs0,
        # yv, ya,
        # Elv, Erv, Era, Ela,
        # Emaxlv, Emaxrv,
        # Vlv0, Vrv0, Vla0, Vra0,
        # Fra, Frv, Fla, Flv,
        # Pra, Prv, Pla, Plv,
        # Vra, Vrv, Vla, Vlv,
        # Frv_sm, COmod, SV,
        # initial conditions
        HRa, HRv, Vra, Vrv, Vla, Vlv, Tsv, Tsa, n, m, tRwave, tPwave,
        # parameters
        Ts1v, Ts1a, Ts2, offv,
        Vlvd0, Vlvs0, Vrvd0, Vrvs0, Vlad0, Vlas0, Vrad0, Vras0,
        Rra, Rla, Rlv, Rrv,
        PRint, KElv, KErv,
        Emaxlv1, Eminlv, Emaxrv1, Eminrv,
        EDVLV, EDVRV,
        Emaxra, Eminra, Emaxla, Eminla,

        # systemic
        # Rsa, Rvc,
        # Paop, Paod, Psa_a, Psa_p, Psa, Psap, Psc, Psv, Pvc,
        # Faop, Faod, Fsa, Fsap, Fsc, Fsv, Fvc, Fcrb,
        # Vaop, Vaod, Vsa, Vsap, Vsc, Vsv, Vvc,
        # Kv, AOFmod,
        # ABPshift, ABPfol, MAPmeas,
        # MAPmod, COmea,
        # initial conditions
        MAPmeas0, Faop, Faod, Frv_sm, Vaop, Vaod, Vsa, Vsap, Vsc, Vsv, Vvc, Paop, AOFmod, ABPfol, COmea,
        # parameters
        KCOMAP, Raop, Rtaop, Rcrb, Raod, Rtaod, Rsap, Rsc, Rsv,
        Caop, Caod, Csap, Csc,
        Vaop0, Vaod0, Vsap0, Vsc0,
        Laop, Laod,
        Kc, Do, Vsa0, Vsa_max, Kp1, Kp2, Kr, Rsa0, tau_p, Ksv, Kv1, Vmax_sv, D2, K1, K2, KR, R0, Vvc0, Vmax_vc, Vmin_vc,
        tauCO, Kxp, Kxv, Kxv1, Kxp1, tauMAP, tauABP,

        # pulmonary
        # Ppap, Ppad, Ppa, Ppc, Ppv,
        # Vpap, Vpad, Vpa, Vpc, Vpv,
        # Fpap, Fpad, Fps, Fpa, Fpc, Fpv,
        # initial conditions
        Vpap, Vpad, Vpa, Vpc, Vpv, Fpap, Fpad,
        # parameters
        Rtpap, Rtpad, Rpap, Rpad, Rps, Rpa, Rpc, Rpv,
        Cpap, Cpad, Cpa, Cpc, Cpv,
        Vpap0, Vpad0, Vpa0, Vpc0, Vpv0,
        Lpap, Lpad,

        # coronary
        # Pcorintrac, Pcorcapc, Pcorvnc,
        # Pcorepi, Pcorintra, Pcorcap, Pcorvn,
        # Vcorepi, Vcorintra, Vcorcap, Vcorvn, Vcorcirc,
        # Fcorepi, Fcorintra, Fcorcap, Fcorvn,
        # initial conditions
        Vcorepi, Vcorintra, Vcorcap, Vcorvn,
        # parameters
        Rcorepi, Rcorintra, Rcorcap, Rcorvn,
        Ccorepi, Ccorintra, Ccorcap, Ccorvn,
        Vcorepi0, Vcorintra0, Vcorcap0, Vcorvn0,

        # baroreceptor
        # Nbr, Nbr_t, N_con, f_con, N_vaso, f_vaso, b_vaso, af_con, af_con2,
        # initial conditions
        Nbr, Nbr_t, N_con, N_vaso, af_con2,
        # parameters
        a, a1, a2, K,
        K_con, T_con, l_con, a_con, b_con, tau_con, No_con,
        K_vaso, T_vaso, l_vaso, a_vaso, tau_vaso, No_vaso,
        amin, bmin, Ka, Kb,

        # blood
        Vheart, VSysArt, VSysVen, VPulArt):

    HRa, HRv, Vra, Vrv, Vla, Vlv, Tsv, Tsa, Vvarlvs0, Vvarrvs0, n, m, tRwave, tPwave,\
    MAPmeas, Faop, Faod, Frv_sm, Vaop, Vaod, Vsa, Vsap, Vsc, Vsv, Vvc, Paop, AOFmod, ABPfol, COmea,\
    Vpap, Vpad, Vpa, Vpc, Vpv, Fpap, Fpad,\
    Vcorepi, Vcorintra, Vcorcap, Vcorvn,\
    Nbr, Nbr_t, N_con, N_vaso, af_con2 = y

    f_con = a_con + b_con / (np.exp(tau_con * (N_con - No_con)) + 1.0)
    af_con = amin + (Ka * f_con)

    # heart
    resultA = _trigger_A(t, tHB, HP, PRint, offv, Ts1a, Ts2)
    if resultA is not None:
        HRa, Tsa, tPwave = resultA

    resultB = _trigger_B(t, tHB, HP, offv, Ts1v, Ts2, n,
                         Vlv, Vlvd0, EDVLV, Vlvs0,
                         Vrv, Vrvd0, EDVRV, Vrvs0,
                         af_con)
    if resultB is not None:
        HRv, Tsv, tRwave, Vvarlvs0, Vvarrvs0, af_con2 = resultB

    # ----------- heart ----------- #
    ta_rel = t_rel(t, tPwave)
    tv_rel = t_rel(t, tRwave)
    # activation functions
    ya = _yi(ta_rel, Tsa)
    yv = _yi(tv_rel, Tsv)
    # elastances
    Era = _ei(Emaxra, Eminra, ya)
    Erv = _ei(Emaxra, Eminra, yv)
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
    Plv = _pv(Elv, Vlv, Vlv0, af_con2, Kxp, Kxv)
    # flows
    Fra = _fi(Pra, Prv, Rra)
    Frv = _fi(Prv, Ppap, Rrv)
    Fla = _fi(Pla, Plv, Rla)
    Flv = _fi(Plv, Paop, Rlv)

    # ----------- systemic circulation ----------- #
    # misc
    COmod = _cardiac_output(Frv_sm)
    SV = _stroke_volume(COmod, HR)
    ABPshift = _arterial_blood_pressure(ABPmeas, t, offv)
    Kv = _kv(Kv1, Ksv)
    MAPmod = _map(Rtaod, Rcrb, AOFmod, Faod,
                  Vaod, Vaod0, Caod, Kxp, Kxv, Pvc)

    # pressures
    Paod = _aortic_afterload(ABPshift)
    Psap = _psap(Vsap, Vsap0, Csap, Kxp, Kxv)
    Psa_a = _psa_a(Kc, Vsa, Vsa0, Do)
    Psa_p = _psa_p(Kp1, Kp2, tau_p, Vsa, Vsa0)
    Psa = _psa(f_vaso, Psa_a, Psa_p)
    Psc = _psc(Vsc, Vsc0, Csc, Kxp, Kxv)
    Psv = _psv(Kv, Vmax_sv, Vsv)
    Pvc = _pvc(Vvc, Vvc0, Vmin_vc, K1, K2, D2, Kxp, Kxv)

    # flows
    Fcrb = _forward_flow(MAPmod, Pvc, Rcrb)
    Fsap = _forward_flow(Psap, Psa, Rsap)
    Fsa = _forward_flow(Psa, Psc, Rsa)
    Fsc = _forward_flow(Psc, Psv, Rsc)
    Fsv = _forward_flow(Psv, Pvc, Rsv)
    Fvc = _forward_flow(Pvc, Pra, Rvc)

    # differential equations
    d_ABPfol_dt = _abp_change(ABPshift, ABPfol, tauABP)
    d_AOFmod_dt = _aortic_flow_change(MAPmeas, MAPmod, KCOMAP)
    d_Vaop_dt = _vaop_change(Paop, Vaop, Vaop0, Caop, Rtaop)
    d_Vaod_dt = _v_change(AOFmod, Faod + Fcrb)
    d_Vsa_dt = _v_change(Fsap, Fsa)
    d_Vsap_dt = _v_change(Faod, Fsap)
    d_Vsc_dt = _v_change(Fsa, Fsc)
    d_Vsv_dt = _v_change(Fsc, Fsv)
    d_Vvc_dt = _v_change(Fsv + Fcrb, Fvc)
    d_Frv_sm_dt = _pulmonary_valve_flow_change(Frv, Frv_sm, tauC0)
    d_Faop_dt = _forward_flow(Paop, Faop * Raop + Paod, Laop)
    d_Faod_dt = _forward_flow(MAPmod, Faod * Raod - Psap, Laod)
    d_Paop_dt = _pressure_aop_change(Flv, d_Vaop_dt, Faop, Fcorepi, Ccorepi)

    # ----------- pulmonary circulation ----------- #
    # pressures
    Ppap = _ppap(Prv, Ppap, Rtpap, Rrv, Fpap, Vpap, Vpap0, Cpap, Kxp, Kxv)
    Ppad = _ppad(Vpad, Vpad0, Kxp, Kxv, Fpap, Rtpad, Fpad, Cpad)
    Ppa = _pp(Vpa, Vpa0, Cpa, Kxp, Kxv)
    Ppc = _pp(Vpc, Vpc0, Cpc, Kxp, Kxv)
    Ppv = _pp(Vpv, Vpv0, Cpv, Kxp, Kxv)

    # flows
    Fps = _fp(Ppa, Ppv, Rps)
    Fpa = _fp(Ppa, Ppc, Rpa)
    Fpc = _fp(Ppc, Ppv, Rpc)
    Fpv = _fp(Ppv, Pla, Rpv)

    # differential equations
    d_Vpad_dt = _vp(Fpap, Fpad)
    d_Vpap_dt = _vp(Frv, Fpap)
    d_Vpa_dt = _vp(Fpad, Fps + Fpa)
    d_Vpc_dt = _vp(Fpa, Fpc)
    d_Vpv_dt = _vp(Fpc + Fps, Fpv)
    d_Fpap_dt = _fpa(Ppap, Ppad + Fpap*Rpap, Lpap)
    d_Fpad_dt = _fpa(Ppad, Ppa + Fpad*Rpad, Lpad)

    # ----------- coronary circulation ----------- #
    # pressures
    Pcorepi = _p_eq(Paop)
    Pcorintra = _pcor(Vcorintra, Vcorintra0, Ccorintra, Kxp1, Kxv1)
    Pcorcap = _pcor(Vcorcap, Vcorcap0, Ccorcap, Kxp1, Kxv1)
    Pcorvn = _pcor(Vcorvn, Vcorvn0, Ccorvn, Kxp, Kxv)
    Pcorintrac = _pcorc(Pcorintra, Plv)
    Pcorcapc = _pcorc(Pcorcap, Plv)
    Pcorvnc = _p_eq(Pcorvn)

    # flows
    Fcorepi = _fcor(Pcorepi, Pcorintrac, Rcorepi)
    Fcorintra = _fcor(Pcorintrac, Pcorcapc, Rcorintra)
    Fcorcap = _fcor(Pcorcapc, Pcorvnc, Rcorcap)
    Fcorvn = _fcor(Pcorvnc, Pra, Rcorvn)

    # differential equations
    d_Vcorepi_dt = _vcor(Flv, d_Vaop_dt + Faop + Fcorepi)
    d_Vcorintra_dt = _vcor(Fcorepi, Fcorintra)
    d_Vcorcap_dt = _vcor(Fcorintra, Fcorcap)
    d_Vcorvn_dt = _vcor(Fcorcap, Fcorvn)

    # ----------- baroreceptor ----------- #
    # Equation 2 order!

    # ----------- blood ----------- #
    tbv = _total_blood_volume(Vheart, VPulArt, Vpc, Vpv, VSysArt, Vsc, VSysVen)
    hv = _heart_volume(Vra, Vrv, Vla, Vlv, Vcorcirc)
    cv = _coronary_volume(Vcorepi, Vcorintra, Vcorcap, Vcorvn)
    sav = _systemic_arterial_volume(Vaop, Vaod, Vsap, Vsa)
    svv = _systemic_venous_volume(Vsv, Vvc)
    pav = _pulmonary_arterial_volume(Vpap, Vpad, Vpa)

    # heart
    d_Vra_dt = _volume_change(Fvc + Fcorvn, Fra)
    d_Vrv_dt = _volume_change(Fra, Frv)
    d_Vla_dt = _volume_change(Fpv, Fla)
    d_Vlv_dt = _volume_change(Fla, Flv)

    dx_dt = np.array([
        d_Vra_dt, d_Vrv_dt, d_Vla_dt, d_Vlv_dt,
        d_ABPfol_dt, d_AOFmod_dt, d_Vaop_dt, d_Vaod_dt, d_Vsa_dt, d_Vsap_dt, d_Vsc_dt, d_Vsv_dt, d_Vvc_dt, d_Frv_sm_dt, d_Faop_dt, d_Faod_dt, d_Paop_dt,
        d_Vpad_dt, d_Vpap_dt, d_Vpa_dt, d_Vpc_dt, d_Vpv_dt, d_Fpap_dt, d_Fpad_dt,
        d_Vcorepi_dt, d_Vcorintra_dt, d_Vcorcap_dt, d_Vcorvn_dt
    ])
    return dx_dt


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
    tRwave = np.min(tmeas) - params.loc["offv", "value"]
    tPwave = np.min(tmeas) - params.loc["PRint", "value"] - params.loc["offv", "value"]

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

    # initial condition for the ODE solver
    y0 = np.array([
        HRa, HRv, Vra, Vrv, Vla, Vlv, Tsv, Tsa, Vvarlvs0, Vvarrvs0, n, m, tRwave, tPwave,
        MAPmeas, Faop, Faod, Frv_sm, Vaop, Vaod, Vsa, Vsap, Vsc, Vsv, Vvc, Paop, AOFmod, ABPfol, COmea,
        Vpap, Vpad, Vpa, Vpc, Vpv, Fpap, Fpad,
        Vcorepi, Vcorintra, Vcorcap, Vcorvn,
        Nbr, Nbr_t, N_con, N_vaso, af_con2,
    ])

    ODE_args = (
        x, tHB, HP, tmeas, ABPmeas,
    )
    sol = solve_ivp(fun=ODE, t_span=[np.min(tmeas), np.max(tmeas)], y0=y0,
                    args=ODE_args, t_eval=tmeas,
                    method="RK23")

    return
