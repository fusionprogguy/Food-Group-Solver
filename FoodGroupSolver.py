# Food Group Solver
#
# This program tries to minimise the risk of different diseases by changing the level of intake of different
# food groups according to systematic reviews and meta-analysis of prospective studies
# See this google sheet called

# 'Food Group - Risk Calculator'
# https://docs.google.com/spreadsheets/d/1Ljs76W2wtF9ZzI7mlBjaVQntj0Gtr6Ni5A-qsqy-Bp0/edit#gid=0

# Originally inspired by Nick Hiebert's video and articles on how to build a healthy diet
# https://www.patreon.com/posts/vlog-4-building-53314386
# https://www.patreon.com/posts/tl-dr-healthy-53572583

# The data for each disease and each food group were obtained from screen shots of the graphs in the papers in Nick's articles above.
# Data was captured using WebPlotDigitizer https://automeris.io/WebPlotDigitizer/ and equations were found using Eureqa https://www.datarobot.com/nutonian/

from numpy import *
from scipy import stats
from scipy import optimize
from random import randint

import matplotlib.pyplot as plt
import numpy as np

def rsquared(x, y):
    return np.corrcoef(x, y)[0, 1]**2

def risk(risk_category, grams):
    # print(grams)
    g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12 = grams
    f1 = f2 = f3 = f4 = f5 = f6 = f7 = f8 = f9 = f10 = f11 = f12 = None

    if risk_category == 'Mortality':
        # Whole grains, <=110g
        if 0 <= g1 <= 110:
            f1 = 1.00000580990638 + 0.0000238899850719038 * g1 ** 2 - 0.00374056674963493 * g1 - 0.0000000890378843780931 * g1 ** 3

        # Refined grains, <=150g
        if 0 <= g2 <= 150:
            f2 = 0.985636361315474 + 0.0000185026832861545 * g2 ** 2 + 0.0143775300500745 * exp(
            -0.0000192653522636716 * g2 ** 3) - 0.00141334351811906 * g2 - 0.0000000460635245795147 * g2 ** 3

        # Vegetables, <=626g
        if 0 <= g3 <= 626:
            f3 = 1.36938867344676 + 0.000000132388589095935 * g3 ** 2 + 3.72247513239414E-13 * g3 ** 4 - 0.000853581602881769 * g3 - 0.369464440678027 * 0.0879027502225784 ** (
                    0.00000198893916572552 * g3 ** 2)

        # Fruits, <=660
        if 0 <= g4 <= 660:
            f4 = 1.0019560148015 + 0.00000292879451127119 * g4 ** 2 + 1.2592678587415E-15 * g4 ** 5 - 0.0009371340002682 * g4 - 0.00000000307365403561747 * g4 ** 3

        # Nuts, <=30
        if 0 <= g5 <= 30:
            f5 = 1.00020987682039 + 0.000825326186481716 * g5 ** 2 + 0.00000000667238935271294 * g5 ** 5 - 0.0219607029518499 * g5 - 0.000000486297357945452 * g5 ** 4

        # Legumes, <=165
        if 0 <= g6 <= 165:
            f6 = 0.999855716315089 + 6.68630308155721E-11 * g6 ** 4 - 0.000399836095940305 * g6 - 0.00000587542916419103 * g6 ** 2

        # Eggs, <=68
        if 0 <= g7 <= 68:
            f7 = 0.999707256448726 + 0.00000190698568573183 * g7 ** 3 - 0.00165259883857732 * g7 - 0.0000000162295041707926 * g7 ** 4

        # Dairy (milk), <=1040
        if 0 <= g8 <= 1040:
            f8 = 0.999991025887482 + 0.000000763238581681405 * g8 ** 2 - 0.000279840453209451 * g8 - 0.000000000337555437446477 * g8 ** 3

        # Fish, <= 250
        if 0 <= g9 <= 250:
            f9 = 1.00000099297196 + 0.0000122538960153683 * g9 ** 2 + 8.07418123444118E-11 * g9 ** 4 - 0.00150028075219509 * g9 - 0.0000000527492780778534 * g9 ** 3

        # Red meat, <=200
        if 0 <= g10 <= 200:
            f10 = 1 + 0.00156812805472349*g10 + 0.00000270380002821566*g10**2

        # Processed meat, <=200
        if 0 <= g11 <= 200:
            f11 = 1.00015688670152 + 0.00490940742143424*g11 + 0.000000246422645637702*g11**3 - 0.000000000511880150476138*g11**4 - 0.0000391239724529405*g11**2

        # Sugar sweetened beverages, <=200
        if 0 <= g12 <= 305:
            f12 = 1.00015688670152 + 0.00490940742143424*g12 + 0.000000246422645637702*g12**3 - 0.000000000511880150476138*g12**4 - 0.0000391239724529405*g12**2

    elif risk_category == 'Obesity':
        # Whole grains, <=110g
        if 0 <= g1 <= 221:
            f1 = 1.0000507675119 + 0.000050767498336214*g1**2 + 0.000000000493243276468996*g1**4 - 0.00488518426603843*g1 - 0.00000026576372246833*g1**3

        # Refined grains, <=150g
        if 0 <= g2 <= 170:
            f2 = 0.999370812052337 + 0.000000886797513048484*g2**3 + 1.82744494474372E-11*g2**5 - 0.00147110638598328*g2 - 0.0000000073612857181474*g2**4

        # Vegetables, <=492g
        if 0 <= g3 <= 492:
            f3 = 0.999848350430688 + 0.0000000113432902532356*g3**3 + 7.36028106705123E-18*g3**6 - 0.000244845332136479*g3 - 1.48468179795039E-11*g3**4 - 0.00000196189477663018*g3**2

        # Fruits, <=660
        if 0 <= g4 <= 280:
            f4 = 1.00085346356918 + 0.0000000104776286111904*g4**3 - 0.000913815389460958*g4 - 1.89664347828468E-11*g4**4

        # Nuts, <=30
        if 0 <= g5 <= 29:
            f5 = 1 + 0.00165436491560647*g5**2 + 0.0000000086265608483863*g5**5 + 0.0720194780061728*g5*0.252832866921258**g5 - 0.0174759216532096*g5 - 0.0000399733746993413*g5**3

        # Legumes, <=165
        if 0 <= g6 <= 165:
            f6 = None

        # Eggs, <=68
        if 0 <= g7 <= 68:
            f7 = None

        # Dairy (milk), <=1040
        if 0 <= g8 <= 718:
            f8 = 1 + 0.000218394224261435*g8 + 6.93564849812325E-16*g8**5 - 6.1467373735473E-22*g8**7 - 0.000000474013012661195*g8**2

        # Fish, <= 250
        if 0 <= g9 <= 86:
            f9 = 1.0000282566737 + 0.0000781762665229839*g9**2 - 0.00780572047322109*g9 - 2.01269620726413E-11*g9**5

        # Red meat, <=200
        if 0 <= g10 <= 192:
            f10 = 0.999997233752634 + 0.000602091052884808*g10 + 0.0000000332925602391651*g10**3 - 9.86163423413665E-11*g10**4

        # Processed meat, <=200
        if 0 <= g11 <= 200:
            f11 = None

        # Sugar sweetened beverages, <=200
        if 0 <= g12 <= 963:
            f12 = 1.0002377520112 + 0.000196223140212869*g12 + 8.59994959848606E-14*g12**4 - 0.000000000129011764070023*g12**3

    elif risk_category == 'Hypertension':
        # Whole grains, <=110g
        if 0 <= g1 <= 92:
            f1 = 1.00000839487255 + 0.00000157693869195293*g1**3 + 9.45220984653665E-11*g1**5 - 0.00422714671099191*g1 - 0.0000000224824140499732*g1**4

        # Refined grains, <=150g
        if 0 <= g2 <= 150:
            f2 = 1.00005950150556 + 0.000000173681689025758*g2**3 + 1.38179328242953E-16*g2**7 - 0.00124308622842593*g2 - 7.91888079505494E-12*g2**5

        # Vegetables, <=492g
        if 0 <= g3 <= 512:
            f3 = 1.0000012067723 + 7.38173813347707E-12*g3**4 + 1.80863521061378E-17*g3**6 - 0.000122094069700771*g3 - 2.20922076838462E-14*g3**5

        # Fruits, <=660
        if 0 <= g4 <= 360:
            f4 = 1.00365358895194 + 0.00000189481674872686*g4**2 - 0.000657022094849668*g4 - 0.00000000175371758439632*g4**3 - 0.00365829691802639*0.946692418883592**g4

        # Nuts, <=30
        if 0 <= g5 <= 37:
            f5 = 0.931658683373132 + 0.00899492979473475*sqrt(g5) + 0.0683253474343096*0.822462562947873**g5 - 0.00374081960262853*g5

        # Legumes, <=165
        if 0 <= g6 <= 71:
            f6 = 1.00012594843061 + 0.000205361227985139*g6 + 0.000000000400007616940544*g6**5 - 2.14968155917602E-12*g6**6 - 0.0000000205148976576964*g6**4

        # Eggs, <=68
        if 0 <= g7 <= 68:
            f7 = None

        # Dairy (milk), <=1040
        if 0 <= g8 <= 798:
            f8 = 1.00102728216276 + 0.000000247933994310627*g8**2 - 0.000345064658146077*g8 - 7.08420724615039E-14*g8**4

        # Fish, <= 250
        if 0 <= g9 <= 156:
            f9 = 0.999935690980483 + 0.00204987783955615*g9 + 0.0000000020732706104165*g9**4 - 5.08376617472772E-12*g9**5 - 0.000000242147056819231*g9**3 - 0.00000387300938657444*g9**2

        # Red meat, <=200
        if 0 <= g10 <= 200:
            f10 = 0.999995239003885 + 0.00156902702239434*g10 + 0.00000263946542770723*g10**2

        # Processed meat, <=200
        if 0 <= g11 <= 39:
            f11 = 0.999950837798111 + 0.00506900719557049*g11 + 0.00000020217817909935*g11**4 - 0.0000000014759722062014*g11**5 - 0.00000803001890549513*g11**3

        # Sugar sweetened beverages, <=200
        if 0 <= g12 <= 456:
            f12 = 1.00002470469514 + 0.000231931045409478*g12 + 0.000000109723482486383*g12**2

    elif risk_category == 'Coronary heart disease':
        # Whole grains, <=110g
        if 0 <= g1 <= 223:
            f1 = 1.00045874331912 + 0.0000199975907171993*g1**2 + 1.25297934670593E-15*g1**6 - 0.00351364848060547*g1 - 0.000000000207819780242973*g1**4

        # Refined grains, <=150g
        if 0 <= g2 <= 220:
            f2 = 1.00019621318546 + 0.0000000268716936354801*g2**3 - 0.000332303129394859*g2 - 4.42891253032498E-21*g2**8

        # Vegetables, <=492g
        if 0 <= g3 <= 549:
            f3 = 0.999968307307653 + 0.00000000422175590229631*g3**3 + 5.3908374442433E-18*g3**6 - 0.000540102790488367*g3 - 7.49072914860986E-12*g3**4

        # Fruits, <=660
        if 0 <= g4 <= 613:
            f4 = 1.13381318595577 + 0.00000241242886039687*g4**2 - 0.00155922095504886*g4 - 1.47771376953214E-12*g4**4 - 0.079022780413264*(0.87707893992527 + 0.0000000145723165173864*g4**3)**(-3.97816899649496)

        # Nuts, <=30
        if 0 <= g5 <= 28:
            f5 = 0.999728098996545 + 0.00164196869051811*g5**2 + 3.09743673861905E-11*g5**7 - 0.0362646804322598*g5 - 0.0000000483030194380726*g5**5

        # Legumes, <=165
        if 0 <= g6 <= 269:
            f6 = 1.00068015060988 + 0.000011518689002265*g6**2 + 4.32551275270271E-16*g6**6 - 0.00217117936477782*g6 - 9.43723494091897E-11*g6**4

        # Eggs, <=68
        if 0 <= g7 <= 75:
            f7 = 1.00000419395789 + 0.000276907388715002*sqrt(g7) + 0.00000956103185622741*g7**2 - 0.0000000469769556746469*g7**3

        # Dairy (milk), <=1040
        if 0 <= g8 <= 703:
            f8 = 0.999960641464613 + 0.00000000131375353919311*g8**3 + 4.02609146523967E-21*g8**7 - 0.000278071522207741*g8 - 4.58765948674084E-18*g8**6

        # Fish, <= 250
        if 0 <= g9 <= 317:
            f9 = 0.945109863241359 + 2.34819521559233E-13*g9**4 + 0.0544254002196295*g9**(-0.0146771655195789*g9) - 0.000372978651365398*g9

        # Red meat, <=200
        if 0 <= g10 <= 100:
            f10 = 1 + 0.00000107877854879899*g10**3 - 0.00128860386567646*g10 - 0.00000000532859003128846*g10**4 - 0.000027595184894278*g10**2

        # Processed meat, <=200
        if 0 <= g11 <= 33:
            f11 = 0.999742525315079 + 0.00994530498234875*g11 + 0.00000457822018822179*g11**3 - 0.000353844884793033*g11**2

        # Sugar sweetened beverages, <=200
        if 0 <= g12 <= 650:
            f12 = 1.00003460032796 + 0.00050156299128057*g12 + 0.000000472280449363417*g12**2 - 0.000000000168024741983963*g12**3

    elif risk_category == 'Stroke':
        # Whole grains, <=110g
        if 0 <= g1 <= 688:
            f1 = 0.810689608671801 + 0.000303540959186707*g1 + 0.190700456910961*0.991151967206131**g1 + 0.000000019319202827936*0.991151967206131**g1*g1**3

        # Refined grains, <=150g
        if 0 <= g2 <= 312:
            f2 = 1.00039399940037 + 0.000000260679238179467*g2**2 - 0.000145761704124746*g2 - 6.58820576349195E-13*g2**4

        # Vegetables, <=492g
        if 0 <= g3 <= 407:
            f3 = 1.00016574776358 + 0.0000000250769907006448*g3**3 + 7.71737893026194E-14*g3**5 - 0.00101592534661781*g3 - 8.086957575042E-11*g3**4

        # Fruits, <=660
        if 0 <= g4 <= 423:
            f4 = 1.0018680303821 + 0.000011006234493039*g4**2 + 1.5562871894236E-14*g4**5 - 0.00255112776940949*g4 - 0.0000000171392032789185*g4**3

        # Nuts, <=30
        if 0 <= g5 <= 30:
            f5 = 1.00099312875435 + 0.000902764274951303*g5**2 - 0.0133568189698753*g5 - 0.0000122141905407157*g5**3

        # Legumes, <=165
        if 0 <= g6 <= 79:
            f6 = 0.999999765613346 + 0.00095139637535958*g6 + 0.0000000918053912528551*g6**3 + 1.20571019919927E+21*g6*0.0000125419822022436**g6 - 0.00181302750217015*sqrt(g6) - 0.0000206862546753395*g6**2

        # Eggs, <=68
        if 0 <= g7 <= 75:
            f7 = 1.00049989350103 + 0.0000346899919357074*g7**2 - 0.00193005298214417*g7 - 0.00000000133660261910152*g7**4

        # Dairy (milk), <=1040
        if 0 <= g8 <= 1004:
            f8 = 1.00016683478597 + 0.000000000744687416673434*g8**3 + 3.57806476720974E-16*g8**5 - 0.000189596499732402*g8 - 9.46337984825463E-13*g8**4

        # Fish, <= 250
        if 0 <= g9 <= 126:
            f9 = 1.00401625799537 + 0.0000442744032278346*g9**2 + 3.00939350364819E-12*g9**5 - 0.00334833739527139*g9 - 0.000000252917933276324*g9**3 - 0.00401354788015853*0.760051485211053**g9

        # Red meat, <=200
        if 0 <= g10 <= 195:
            f10 = 1.00001650799274 + 0.000996845007109223*g10 + 0.00000161653824559303*g10**2 - 0.00000000192214970331105*g10**3

        # Processed meat, <=200
        if 0 <= g11 <= 84:
            f11 = 0.999208246731238 + 0.00538441288935898*g11 + 0.0000000384613303689653*g11**4 - 0.000000000179099939230268*g11**5 - 0.0000024400284334768*g11**3

        # Sugar sweetened beverages, <=200
        if 0 <= g12 <= 624:
            f12 = 1.00004416212234 + 0.000233457185414672*g12 + 0.000000185146656452008*g12**2 - 1.02622727140244E-13*g12**4

    elif risk_category == 'Breast Cancer':
        # Whole grains, <=110g
        if 0 <= g1 <= 688:
            f1 = None

        # Refined grains, <=150g
        if 0 <= g2 <= 244:
            f2 = 0.999647784427458 + 0.000000166694062503251*g2**3 + 1.08713384881896E-12*g2**5 - 0.000699253060068376*g2 - 0.000000000738341463966824*g2**4 - 0.00000994065167271397*g2**2

        # Vegetables, <=492g
        if 0 <= g3 <= 499:
            f3 = 1.01511584428115 + 0.0000015186792841574*g3**2 + 8.69589871944779E-16*g3**5 - 0.000512594578597604*g3 - 0.00000000168767912276405*g3**3 - 0.0151697727479217*0.972167217399176**g3

        # Fruits, <=660
        if 0 <= g4 <= 459:
            f4 = 1.00006340105495 + 0.00000000720966272977647*g4**3 + 7.94592608012118E-15*g4**5 - 0.000182781417604044*g4 - 1.33157255770679E-11*g4**4 - 0.000000894226320882383*g4**2

        # Nuts, <=30
        if 0 <= g5 <= 30:
            f5 = None

        # Legumes, <=165
        if 0 <= g6 <= 79:
            f6 = None

        # Eggs, <=68
        if 0 <= g7 <= 43:
            f7 = 0.999964288907252 + 0.00211373128406166*g7 + 0.00581152260341498*g7*0.000016902179217457**(0.01182312980931*g7) + 0.00145483201246028*g7**2*0.000016902179217457**(0.01182312980931*g7) - 0.000000576478821976537*g7**3

        # Dairy (milk), <=1040
        if 0 <= g8 <= 819:
            f8 =  1.00322826507111 + 0.0000000664767527395768*g8**2 + 0.000073763722326998*g8*0.0000354349549695938**(0.00000000472207324027457*g8**3) + 0.0000522585645963408*g8*0.0000354349549695938**(0.00000000236103662013729*g8**3) - 0.000118400064108107*g8 - 0.00322826271007215*0.000073763722326998**g8

        # Fish, <= 250
        if 0 <= g9 <= 819:
            f9 = 1.00010797228407 + 0.00134496071008393*g9 + 0.000159292442456716*g9**2 + 0.00195053292245334*g9*0.000259456009125647**(0.0141502139014468*g9) - 0.0578172708592872*0.000135749756567022**(0.000259456009125647**(0.0141502139014468*g9)) - 0.000157036190219262*g9**2*0.000135749756567022**(0.000259456009125647**(0.0141502139014468*g9))

        # Red meat, <=200
        if 0 <= g10 <= 151:
            f10 = 0.994888397567691 + 0.000000142760123943435*g10**3 + 0.0026604091075011*sqrt(3.68718917416408 + g10**2) - 1.20623098950862E-12*g10**5 - 0.0000298546670448722*g10**2

        # Processed meat, <=200
        if 0 <= g11 <= 56:
            f11 = 0.999965782329385 + 0.00818021596278548*g11 + 0.000039104905338513*g11**2 + 0.000000207668790492212*g11**3 - 0.466304655775947*g11*sqrt(0.00000703645687717201*g11)

        # Sugar sweetened beverages, <=200
        if 0 <= g12 <= 624:
            f12 = None

    final = list(filter(None, [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]))  # Remove any None values
    return final

def f(grams):
    # print(array(risk(risk_category, grams)))
    return abs(sum(array(risk(risk_category, grams))**2)-0)

food_groups = ['Whole grains', 'Refined grains', 'Vegetables', 'Fruits', 'Nuts', 'Legumes', 'Eggs', 'Dairy', 'Fish', 'Red meat', 'Processed meat', 'Sugar sweetened beverages']
risk_groups = ['Mortality', 'Obesity', 'Hypertension', 'Coronary heart disease', 'Stroke', 'Breast Cancer']

risk_category = 'Hypertension'  # Mortality, Obesity, Hypertension, Coronary heart disease, Stroke, Breast Cancer
random_tries = 1000000

print('\nSolution')

smallest_y_risk = 2  # Make this number the highest risk to start with
g_intake_range = [(750, 1000), (1000, 1250), (1250, 1500), (1500, 1750), (1750, 2000), (2000, 2250), (2250, 2500), (2500, 2750), (2750, 3000)]


### Mathematical Attempt To Find the Minimum Risk

# Unfortunately it gives solutions with negative numbers. A better mathematical solution is required.

g_vars = [[None, 2]] * (len(g_intake_range))
g_smallest_x_risk = []

print('Vars')
print(g_vars)

for g in range(10):
    print('\nMath Solution Test', g)
    # Generate random values for grams for each food group
    Whole_grains = random.randint(75, 221)  # <=110g
    Refined_grains = random.randint(0, 90)  # <=150g
    Vegetables = random.randint(150, 550)  # <=626g
    Fruits = random.randint(200, 350)  # <=660
    Nuts = random.randint(10, 37)  # <=30
    Legumes = random.randint(60, 165)  # <=165
    Eggs = random.randint(0, 50)  # <=68
    Dairy = random.randint(150, 350)  # <=1040
    Fish = random.randint(0, 250)  # <= 250
    Red_meat = random.randint(0, 65)  # <=200
    Processed_meat = random.randint(0, 12)  # <=200
    SSB = random.randint(0, 12)  # <=200

    initial_grams_guess = [Whole_grains, Refined_grains, Vegetables, Fruits, Nuts, Legumes, Eggs, Dairy, Fish, Red_meat, Processed_meat, SSB]

    # Show solution
    solution = optimize.fmin(f, initial_grams_guess)  # Needs initial guess for the solution. It's very sensitive to initial values.
    round_solution_x = [int('{:.0f}'.format(f_grams)) for f_grams in list(solution)]  # Hide decimals

    # print('Check All', all(x >= 0 for x in round_solution_x))

    # Show only positive solutions
    if all(x >= 0 for x in list(round_solution_x)):

        # print(list(solution))
        # print(round_solution_x)  # Hide decimals
        # print(risk(risk_category, round_solution_x), risk_solution_y)
        risk_solution_y = average(risk(risk_category, round_solution_x))

        hist_range = [(g1, g2) for g1, g2 in g_intake_range if (g1 <= sum(round_solution_x) < g2)]      # eg [(1750, 2000)]
        # print('Range', sum(round_solution_x), hist_range)

        if len(hist_range) == 1:
            index = g_intake_range.index(hist_range[0])
            grams_sum = sum(round_solution_x)
            # print('Item', index)

            if all([x >= 0 for x in list(round_solution_x)]):
                if g_vars[index] == [None, 2]:
                    g_vars[index] = [round_solution_x, grams_sum, risk_solution_y]
                else:
                    # update smallest value
                    if risk_solution_y < smallest_y_risk:
                        smallest_x_risk = round_solution_x
                        smallest_y_risk = risk_solution_y
                        # print(smallest_x_risk, grams_sum, smallest_y_risk)

                        # replace old value with new value if new value is smaller
                        if smallest_y_risk < g_vars[index][2]:
                            # print('Throw out old', g_vars[index], [smallest_x_risk, grams_sum, smallest_y_risk])
                            g_vars[index] = [smallest_x_risk, grams_sum, smallest_y_risk]

                            # Save the smallest set of grams for the food groups

    smallest_x_risk = round_solution_x


print(smallest_x_risk, grams_sum, smallest_y_risk)

# Replace negative numbers with 1
# smallest_x_risk = [x if x >= 0 else 1 for x in round_solution_x]    # [x+1 for x in l if x >= 45 else x+5]

### Model Simulation using Random Numbers Only

for risk_category in risk_groups:

    '''
    print('g_vars')
    for g in g_vars:
        print(g)
    '''

    g_vars = [[None, 2]] * (len(g_intake_range))  # Create initial array with default values.

    x_array = []
    y_array = []

    print('\nTesting range', smallest_x_risk, smallest_y_risk)
    smallest_y_risk = 2  # risk_solution_y
    smallest_x_risk = []

    count = 0
    for g in range(1, random_tries):
        # Generate random values for grams for each food group
        Whole_grains = random.randint(75, 221)  # <=110g
        Refined_grains = random.randint(0, 90)  # <=150g
        Vegetables = random.randint(150, 550)  # <=626g
        Fruits = random.randint(200, 350)  # <=660
        Nuts = random.randint(10, 37)  # <=30
        Legumes = random.randint(60, 165)  # <=165
        Eggs = random.randint(0, 50)  # <=68
        Dairy = random.randint(150, 350)  # <=1040
        Fish = random.randint(0, 250)  # <= 250
        Red_meat = random.randint(0, 65)  # <=200
        Processed_meat = random.randint(0, 12)  # <=200
        SSB = random.randint(0, 12)  # <=200

        round_solution_x = [Whole_grains, Refined_grains, Vegetables, Fruits, Nuts, Legumes, Eggs, Dairy, Fish, Red_meat, Processed_meat, SSB]

        # Show solution
        # solution = optimize.fmin(f, initial_grams_guess)
        # round_solution_x = [int('{:.0f}'.format(f_grams)) for f_grams in list(solution)]  # Hide decimals

        # print('Check All', all(x >= 0 for x in round_solution_x))

        # Show only positive solutions
        # if all(x >= 0 for x in list(round_solution_x)):
        # print(list(solution))
        # print(round_solution_x)  # Hide decimals
        grams_sum = sum(round_solution_x)

        risk_solution_y = average(risk(risk_category, round_solution_x))
        # print(risk(risk_category, round_solution_x), risk_solution_y)

        hist_range = [(g1, g2) for g1, g2 in g_intake_range if (g1 <= sum(round_solution_x) < g2)]     # eg [(1750, 2000)]

        x_array.append(grams_sum)
        y_array.append(risk_solution_y)

        # print('Range', sum(round_solution_x), hist_range)
        if len(hist_range) == 1:
            index = g_intake_range.index(hist_range[0])

            # print('Item', index)
            if all([x >= 0 for x in list(round_solution_x)]):
                if g_vars[index] == [None, 2]:
                    g_vars[index] = [round_solution_x, grams_sum, risk_solution_y]
                else:
                    # update smallest value
                    if risk_solution_y < smallest_y_risk:
                        smallest_x_risk = round_solution_x
                        smallest_y_risk = risk_solution_y
                        # print(smallest_x_risk, grams_sum, smallest_y_risk)

                        # replace old value with new value if new value is smaller
                        if smallest_y_risk < g_vars[index][2]:
                            # print('Throw out old', g_vars[index], [smallest_x_risk, grams_sum, smallest_y_risk])
                            g_vars[index] = [smallest_x_risk, grams_sum, smallest_y_risk]

                            # Save the smallest set of grams for the food groups

    print(smallest_x_risk, grams_sum, smallest_y_risk)

    print('\n' + risk_category)
    print('\nRandom Solutions across Ranges ' + risk_category)
    print(g_intake_range)
    print(food_groups)
    for g in g_vars:
        print(g)

    print('\nBest Random Solution ' + risk_category)
    print(smallest_x_risk, sum(smallest_x_risk), smallest_y_risk)

    # Show scatter plot with linear fit and equation
    x = np.array(x_array)
    y = np.array(y_array)
    m, b = np.polyfit(x, y, 1)       # m = slope, b = intercept
    r2 = round(rsquared(x, y), 7)
    equation = 'y = ' + str(round(m, 7)) + 'x' ' + ' + str(round(b, 7)) + '  r^2 = ' + str(r2)
    print(equation)

    plt.figure()
    plt.scatter(x_array, y_array, s=1)
    plt.plot(x, m*x + b, color='r')
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes)
    plt.title(risk_category)
    plt.ylabel('Risk ' + risk_category)
    plt.xlabel('kcal')
    plt.savefig(risk_category + ' Risk.png')
    # plt.show()

for risk_category in risk_groups:  # eg 'Mortality', 'Obesity' ...
    for food_group in food_groups:  # eg 'Whole grains', 'Refined grains' ...
        print('\n' + food_group)
        print(risk_category + ': ' + food_group + ' across intakes of g')
        x_array = []
        y_array = []

        for g in range(0, 500):
            fill_list = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            index = food_groups.index(food_group)  # find the food group location in the array
            fill_list[index] = g  # set the input value for grams for the food group
            risk_solution_y = average(risk(risk_category, fill_list))  # calculate the y value
            # print(fill_list, risk_solution_y)

            x_array.append(g)
            y_array.append(risk_solution_y)

        plt.figure()
        plt.scatter(x_array, y_array, s=1)
        plt.title(risk_category + ': ' + food_group)
        plt.ylabel('Risk ' + risk_category)
        plt.xlabel('grams')
        plt.savefig(food_group + ' ' + risk_category + ' Risk.png')
