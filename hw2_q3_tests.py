from hw2 import *

# Each test case contains: mus, sigmas, weights
# 'desc' explains what the test case is specifically checking.
gmm_test_cases = [
    {
        "id": 1, "k": 2, "desc": "Simple & Distant",
        "mus": [ -20.0, 20.0 ], "sigmas": [ 1.0, 1.0 ], "ws": [ 0.5, 0.5 ]
    },
    {
        "id": 2, "k": 2, "desc": "Uneven Weights (90/10 split)",
        "mus": [ 0.0, 10.0 ], "sigmas": [ 0.5, 0.5 ], "ws": [ 0.9, 0.1 ]
    },
    {
        "id": 3, "k": 3, "desc": "Varying Widths",
        "mus": [ -10.0, 0.0, 10.0 ], "sigmas": [ 0.2, 2.0, 0.2 ], "ws": [ 0.33, 0.34, 0.33 ]
    },
    {
        "id": 4, "k": 3, "desc": "Close Proximity (Testing Resolution)",
        "mus": [ 0.0, 2.0, 4.0 ], "sigmas": [ 0.4, 0.4, 0.4 ], "ws": [ 0.3, 0.4, 0.3 ]
    },
    {
        "id": 5, "k": 4, "desc": "Symmetric Clusters",
        "mus": [ -15.0, -5.0, 5.0, 15.0 ], "sigmas": [ 1.0, 1.0, 1.0, 1.0 ], "ws": [ 0.25, 0.25, 0.25, 0.25 ]
    },
    {
        "id": 6, "k": 5, "desc": "The 'Spike' (One extremely narrow component)",
        "mus": [ -10.0, -5.0, 0.0, 5.0, 10.0 ],
        "sigmas": [ 1.0, 1.0, 0.05, 1.0, 1.0 ], "ws": [ 0.2, 0.2, 0.2, 0.2, 0.2 ]
    },
    {
        "id": 7, "k": 2, "desc": "Heavily Overlapping (Stress Test)",
        "mus": [ 0.0, 1.5 ], "sigmas": [ 1.5, 1.5 ], "ws": [ 0.5, 0.5 ]
    },
    {
        "id": 8, "k": 6, "desc": "High K, Large Range",
        "mus": [ -50.0, -30.0, -10.0, 10.0, 30.0, 50.0 ],
        "sigmas": [ 2.0, 1.5, 1.0, 1.0, 1.5, 2.0 ], "ws": [ 0.1, 0.2, 0.2, 0.2, 0.2, 0.1 ]
    },
    {
        "id": 9, "k": 3, "desc": "Minimum Weight Floor (0.05 test)",
        "mus": [ -5.0, 0.0, 5.0 ], "sigmas": [ 1.0, 1.0, 1.0 ], "ws": [ 0.05, 0.90, 0.05 ]
    },
    {
        "id": 10, "k": 4, "desc": "Diverse Scale and Weight",
        "mus": [ -20.0, -18.0, 5.0, 40.0 ], "sigmas": [ 0.3, 0.3, 5.0, 1.2 ], "ws": [ 0.2, 0.2, 0.4, 0.2 ]
    }
]

for case in gmm_test_cases:
    # Last parm (5000) is the requested number of data points to be generated.
    data = q3d(np.array(case['mus']), np.array(case['sigmas']), np.array(case['ws']), 5000)
    mus_nan = np.array(case['mus']).copy()
    sigmas_nan = np.array(case['sigmas']).copy()
    ws_nan = np.array(case['ws']).copy()
    # The expected format for Q3: [1.2, NAN, 1.3]
    # Where 1.2 and 1.3 are the known fixed values and NAN is the value to be estimated
    def inject_nans(arr):
        if len(arr) >=2:
            num_nans = np.random.randint(1, len(arr))
        else:
            num_nans = 1
        indices = np.random.choice(len(arr), size=num_nans, replace=False)
        arr[indices] = np.nan
        return arr
    mus_nan = inject_nans(mus_nan)
    sigmas_nan = inject_nans(sigmas_nan)
    ws_nan = inject_nans(ws_nan)

    mus_res, sigmas_res, ws_res = my_EM(mus_nan, sigmas_nan, ws_nan, data)

    sigmas_res = np.sort(sigmas_res)
    mus_res = np.sort(mus_res)
    ws_res = np.sort(ws_res)

    sigmas_golden = np.sort(np.array(case['sigmas']))
    mus_golden = np.sort(np.array(case['mus']))
    ws_golden = np.sort(np.array(case['ws']))

    mus_check = np.all(np.isclose(mus_res, mus_golden, atol=0.3))
    sigmas_check = np.all(np.isclose(sigmas_res, sigmas_golden, atol=0.1))
    ws_check = np.all(np.isclose(ws_res, ws_golden, atol=0.03))
    if mus_check and sigmas_check and ws_check:
        print("PASS: " + str(case['id']))
    else:
        print("FAIL: " + str(case['id']))
        print(mus_golden)
        print(mus_res)
        print(sigmas_golden)
        print(sigmas_res)
        print(ws_golden)
        print(ws_res)