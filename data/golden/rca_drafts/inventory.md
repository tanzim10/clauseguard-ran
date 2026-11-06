# RCA case inventory (draft)

| Case | Split | Event | Source recording (family) | Dataset label | Window rows (raw) | Baseline / fault samples | Evidence channels | Families |
|---|---|---|---|---|---|---|---|---|
| rca_001 | dev | E09 | Zone_A/Twitch (`family-6aaa4378`) | Co-Channel Interference (Severe) | 5249–5376 | 67 / 61 | UL_SNR, RSRP, UL_BLER | multi_clause |
| rca_002 | dev | E10 | Zone_C/Twitch (`family-128a65d1`) | Co-Channel Interference (Severe) | 5313–5440 | 73 / 55 | UL_SNR, RSRP, UL_BLER | multi_clause |
| rca_003 | dev | E14 | Zone_A/Twitch (`family-6aaa4378`) | High Network Congestion (Gradual Buildup) | 4129–4256 | 66 / 62 | PRB_Utilization_DL, PRB_Utilization_UL, DL_NumberOfPackets, UL_NumberOfPackets | multi_clause |
| rca_004 | dev | E15 | Zone_A/Twitch (`family-6aaa4378`) | High Network Congestion (Sudden Spike) | 833–960 | 57 / 71 | UL_BLER, Estimated_UL_Buffer | multi_clause, difficult |
| rca_005 | dev | E16 | Zone_A/Twitch (`family-6aaa4378`) | High Network Congestion (Sudden Spike) | 3809–3936 | 53 / 75 | UL_BLER, Estimated_UL_Buffer | multi_clause, difficult |
| rca_006 | test | E21 | Zone_A/YouTube (`family-eafe6e67`) | Buffer Overflow (Gradual Buildup) | 4161–4288 | 75 / 53 | PRB_Utilization_DL, PRB_Utilization_UL, DL_NumberOfPackets, UL_NumberOfPackets | multi_clause, difficult |
| rca_007 | test | E23 | Zone_B/YouTube (`family-e6c16ffb`) | Buffer Overflow (Gradual Buildup) | 8673–8800 | 66 / 62 | PRB_Utilization_UL, UL_NumberOfPackets | multi_clause, difficult |
| rca_008 | test | E26 | Zone_B/YouTube (`family-e6c16ffb`) | Co-Channel Interference (Mild) | 321–448 | 55 / 73 | UL_SNR, RSRP, UL_NPRB | multi_clause, difficult |
| rca_009 | test | E28 | Zone_A/YouTube (`family-eafe6e67`) | Co-Channel Interference (Severe) | 1633–1760 | 51 / 77 | UL_SNR, RSRP, UL_NPRB, RX_Bytes | multi_clause |
| rca_010 | test | E29 | Zone_A/YouTube (`family-eafe6e67`) | Co-Channel Interference (Severe) | 8545–8672 | 74 / 54 | UL_SNR, RSRP, UL_NPRB, RX_Bytes | multi_clause |
| rca_011 | test | E36 | Zone_B/YouTube (`family-e6c16ffb`) | High Network Congestion (Gradual Buildup) | 161–288 | 84 / 39 | PRB_Utilization_DL, PRB_Utilization_UL, UL_NumberOfPackets, Estimated_UL_Buffer | multi_clause |
| rca_012 | test | E38 | Zone_B/YouTube (`family-e6c16ffb`) | High Network Congestion (Gradual Buildup) | 3713–3840 | 51 / 77 | PRB_Utilization_DL, PRB_Utilization_UL, UL_NPRB, DL_NumberOfPackets, UL_NumberOfPackets | multi_clause |

**Independence:** 12 cases from 12 distinct anomaly events, but only 4 synthetic base recordings (2 dev, 2 test); recording-family independence is unverified (all 33 recording reviews unresolved).

## Rejected candidates

| Event | Split | Dataset label | Rule failed |
|---|---|---|---|
| E01 | dev | Antenna Failure | rule6: no mechanism clause in corpus |
| E02 | dev | Antenna Failure | rule6: no mechanism clause in corpus |
| E03 | dev | Antenna Failure | rule6: no mechanism clause in corpus |
| E04 | dev | Antenna Failure | rule6: no mechanism clause in corpus |
| E05 | dev | Buffer Overflow (Gradual Buildup) | rule5: agree=['UL_NumberOfPackets'] contra=[] |
| E06 | dev | Buffer Overflow (Gradual Buildup) | rule5b: primary symptom ['PRB_Utilization_DL', 'PRB_Utilization_UL', 'UL_NumberOfPackets'] not observed; agree=['Estimated_UL_Buffer'] |
| E07 | dev | Co-Channel Interference (Mild) | rule3: no window with >=16 pre rows and >=8 anomaly rows |
| E08 | dev | Co-Channel Interference (Mild) | rule5b: primary symptom ['UL_SNR'] not observed; agree=[] |
| E11 | dev | Doppler Shift (Severe) | rule6: no mechanism clause in corpus |
| E12 | dev | Faulty Handover Algorithm (Too Frequent) | rule6: handover signature (0.1-0.3 Hz RSRP/SNR oscillation) not observable in span; single-gNB testbed |
| E13 | dev | Faulty RF Filters (Temporal) | rule6: no mechanism clause in corpus |
| E17 | dev | High Network Congestion (Sudden Spike) | rule5: agree=['Estimated_UL_Buffer'] contra=[] |
| E18 | dev | Resource Allocation Bugs | rule6: weak mechanism support (scheduler describes correct behaviour only) |
| E19 | test | Antenna Failure | rule6: no mechanism clause in corpus |
| E20 | test | Antenna Failure | rule6: no mechanism clause in corpus |
| E24 | test | Co-Channel Interference (Mild) | rule5b: primary symptom ['UL_SNR'] not observed; agree=['RSRP', 'UL_NPRB'] |
| E25 | test | Co-Channel Interference (Mild) | rule5b: primary symptom ['UL_SNR'] not observed; agree=['UL_NPRB'] |
| E30 | test | Doppler Shift (Severe) | rule6: no mechanism clause in corpus |
| E31 | test | Faulty Handover Algorithm (Too Frequent) | rule6: handover signature (0.1-0.3 Hz RSRP/SNR oscillation) not observable in span; single-gNB testbed |
| E32 | test | Faulty Handover Algorithm (Too Frequent) | rule6: handover signature (0.1-0.3 Hz RSRP/SNR oscillation) not observable in span; single-gNB testbed |
| E33 | test | Faulty Handover Algorithm (Too Frequent) | rule3: no window with >=16 pre rows and >=8 anomaly rows |
| E34 | test | Faulty RF Filters (Temporal) | rule6: no mechanism clause in corpus |
| E35 | test | Faulty RF Filters (Temporal) | rule6: no mechanism clause in corpus |
| E39 | test | High Network Congestion (Sudden Spike) | rule5: agree=['UL_NPRB'] contra=[] |
| E40 | test | High Network Congestion (Sudden Spike) | rule5: agree=['UL_NPRB'] contra=[] |
| E41 | test | Resource Allocation Bugs | rule6: weak mechanism support (scheduler describes correct behaviour only) |

Eligible but not picked (rule 7 balance): E27, E22, E37

Real jamming (dev jammer/YouTube, test jammer/File): rejected under rule 2 — whole-window label, no in-window baseline, jammer-on and jammer-off rows indistinguishable.
