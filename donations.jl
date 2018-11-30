using POMDPs
using QMDP
using POMDPModelTools
using ParticleFilters, Distributions
using Random

import StatsBase: countmap

# state, action, observation
mutable struct DonationsPOMDP <: POMDP{Tuple{Int64, Int64, Int64}, Int64, Tuple{Int64, Int64, Int64}}
    win_r::Int64 # winning reward
    total_steps::Int64
    initial_supp::Int64
    initial_budg::Int64
    opp_money::Int64 # opponent's money
    pol_money::Int64 # supported politician's money
    deterministic::Bool
end

DonationsPOMDP() = DonationsPOMDP(100, 10, 45, 100, 100, 100, false)

POMDPs.updater(problem::DonationsPOMDP) = ParticleFilters(problem)

POMDPs.actions(::DonationsPOMDP) = Tuple(0:100)

POMDPs.actions(::DonationsPOMDP, s::Tuple{Int64, Int64, Int64}) = Tuple(0:s[3])

POMDPs.actionindex(::DonationsPOMDP, a::Int64) = a + 1

POMDPs.n_actions(::DonationsPOMDP) = 101

function POMDPs.states(pomdp::DonationsPOMDP)
    ret = []
    for num in 0:pomdp.total_steps
        for vote_per in 0:100
            for money in 0:100
                push!(ret, (num, vote_per, money))
            end
        end
    end
    return ret
end

function POMDPs.stateindex(pomdp::DonationsPOMDP, s::Tuple{Int64, Int64, Int64})
    a = zeros(pomdp.total_steps + 1, 101, 101)
    return LinearIndices(a)[s[1] + 1, s[2] + 1, s[3] + 1]
end 

POMDPs.n_states(pomdp::DonationsPOMDP) = (pomdp.total_steps + 1)*101*101

POMDPs.observations(::DonationsPOMDP) = Tuple(0:100)

POMDPs.obsindex(::DonationsPOMDP, o::Int64) = o + 1

POMDPs.n_observations(::DonationsPOMDP) = 101

POMDPs.discount(p::DonationsPOMDP) = 1

function POMDPs.initialstate(pomdp::DonationsPOMDP, rng::AbstractRNG)
    return (pomdp.total_steps, pomdp.initial_supp, pomdp.initial_budg)
end

function POMDPs.transition(pomdp::DonationsPOMDP, s::Tuple{Int64, Int64, Int64}, a::Int64)
    num_steps = s[1] - 1
    if num_steps <= 0 # end of race, can't do anything
        return SparseCat([s], [1.0])
    end
    a = min(a, s[3]) # cannot give more than you have
    agent_money = s[3] - a
    pomdp.pol_money = pomdp.pol_money + a
    money_percent = 100*pomdp.pol_money/(pomdp.pol_money + pomdp.opp_money)
    if pomdp.deterministic
        return SparseCat([(num_steps, Int(money_percent), agent_money)], [1.0])
    end
    supps = []
    for ratio in vote_money
        support = min(floor(money_percent*ratio), 100)
        push!(supps, support)
    end
    counts = countmap(supps) 
    total = sum(values(counts))
    states = []
    probs = []
    for (key, value) in counts 
        push!(states, (num_steps, Int(key), agent_money))
        push!(probs, value/total)
    end
    return SparseCat(states, probs)
end

function POMDPs.observation(pomdp::DonationsPOMDP, a::Int64, sp::Tuple{Int64, Int64, Int64})
    if sp[1] == 0 # election is over, everything is stable
        return SparseCat([sp], [1.0])
    end
    polls = []
    for ratio in poll_vote
        poll = min(floor(sp[2]*ratio), 100)
        push!(polls, poll)
    end
    counts = countmap(polls)
    total = sum(values(counts))
    states = []
    probs = []
    for (key, value) in counts 
        push!(states, (sp[1], Int(key), sp[3]))
        push!(probs, value/total)
    end
    return SparseCat(states, probs)
end

function POMDPs.reward(pomdp::DonationsPOMDP, s::Tuple{Int64, Int64, Int64}, a::Int64, sp::Tuple{Int64, Int64, Int64})
    r = 0.0
    r -= sp[3] # penalty for what you spent
    if sp[2] > 50
        r += pomdp.win_r * ((pomdp.total_steps-sp[1]+1)/pomdp.total_steps+1)
    end
    return r
end

function POMDPs.reward(pomdp::DonationsPOMDP, s::Tuple{Int64, Int64, Int64}, a::Int64)
    # get expected reward
    r = 0.0
    cats = POMDPs.transition(pomdp, s, a)
    for (sp, prob) in cats
        reward = 0.0
        reward -= sp[3] # penalty for what you spent
        if sp[2] > 50
            reward += pomdp.win_r * ((pomdp.total_steps-sp[1]+1)/pomdp.total_steps+1)
        end
        r += reward*prob
    end
    return r
end

const global vote_money = [0.919262667948769, 0.682449561975545, 0.6879000000000001, 0.7041781387859604, 0.9035446984742327, 0.5806567680235275, 0.6912150479257964, 1.9190789333259015, 1.3169805279720037, 1.4217430746313195, 1.0059395859349398, 0.709937890010264, 1.130357970790027, 0.8123277361317275, 0.6200293045012748, 0.7225894157730538, 0.6774, 1.0400607341878192, 0.69404595039395, 0.9869909910373742, 0.7957537502204713, 1.1275292661486216, 0.744875686149193, 0.8349859202399769, 0.6406000000000001, 0.6825524327120143, 0.6721305462560394, 0.7815000000000001, 0.9640769254851524, 0.6832988484286715, 0.6629260819075317, 0.6279842372459193, 0.7809352992028652, 0.9795999999999999, 1.2251779278689594, 1.0989646528968207, 0.8484, 0.5586094824045625, 0.9926999999999999, 0.8295267103615536, 0.7756000000000001, 0.9197, 0.6165479589022115, 0.942972387790516, 0.602573619814974, 0.9612897718907714, 1.2553861303576983, 0.2144, 1.0447379609063219, 0.6691319141798673, 0.6483964653914221, 1.415669730324187, 1.0657354919591675, 0.7065488204194762, 0.8594440550400216, 0.7551466495834177, 1.1889506507718546, 0.6118, 0.6953080642589771, 0.9833, 0.8828133560633497, 1.075867220593378, 0.6318119979678288, 1.3732219461345199, 0.7191176485059197, 0.9829000000000001, 0.9708162469575438, 0.8727498322561769, 0.6967117736199924, 0.637412320023995, 0.6604676028447671, 1.3823692788466095, 0.6316811538288132, 1.0, 0.9512915476574345, 0.7565999999999999, 0.6152111431569939, 0.737848065172914, 0.7475, 0.9386178888503002, 1.0653771407825854, 0.7309, 0.8387241774981306, 0.8024028244569916, 0.7354180153594504, 0.7143426320083273, 0.8143331387125895, 0.6852362247335323, 0.7931507389728677, 0.8159874659484362, 0.7116535227885172, 0.6818279589288033, 0.6245, 0.7451000000000001, 0.7867000000000001, 0.6858371060567818, 0.7074734680228226, 0.7312000000000001, 0.9749533327106887, 0.7644927095701676, 0.7058741011334365, 0.8642100051203965, 0.7588830566080437, 0.6817922820893044, 1.16663323432458, 0.9355136593390935, 0.7695522464191971, 1.2985179081392118, 0.9515719276736968, 1.8274317882240385, 0.7002999999999999, 1.0452832115998132, 0.618524717264233, 0.680542472966734, 1.0993259229750583, 1.378469004036521, 0.7631100756047272, 0.6761229637313515, 1.0113194305583382, 0.6868000000000001, 0.7552130000987207, 0.9284353685100721, 0.7254, 0.6118038807387826, 0.8209999999999998, 0.8208906265848341, 1.2216027979516153, 0.5957302510076236, 0.8399759377354064, 0.6923719812701629, 0.7283776465440095, 1.145677206940076, 0.640774271691289, 1.60197818481631, 0.8536, 0.8931999999999999, 0.6221626714185944, 0.7746987361067977, 0.0447067389669469, 0.818222262923906, 1.2730146106177451, 0.6612649816014502, 0.7526878526617241, 0.7012486518963555, 0.7281474597968811, 1.0, 1.1792472349478829, 1.9233255172984385, 0.7616380830098665, 0.7510763682205397, 0.7260090054364986, 1.4852374077502117, 0.7287362743181713, 0.8284, 0.6597995000351203, 0.7469379709980799, 0.6173508977270065, 0.9891486923632765, 0.6720999999999999, 0.7697208002075675, 0.766796382929548, 0.67025548883798, 0.7581, 0.8652138334561682, 0.689889736193932, 1.2335769111521973, 0.8651000000000001, 0.5935800377343501, 0.7215, 0.9484999999999999, 0.8963019382889034, 1.0508869284139122, 0.6186609706181263, 0.7386593158179267, 0.6583010296193943, 1.699287222311741, 0.8538, 1.0002426664408453, 1.1332690035710182, 0.7624525442637512, 0.6645617753752767, 0.6818292899586168, 0.9337185123907248, 0.9603790407892129, 0.7355849798229506, 0.6160278957869412, 0.5920693697759073, 1.6105865990809793, 0.893, 1.6650004868446626, 0.9568309091831966, 1.1848748932240505, 0.9979874744552354, 1.3688423830542256, 1.211111014238174, 0.7580412161062289, 0.8665, 0.6163787519899739, 0.7577482491361159, 0.6662824092918356, 0.7593396475589357, 0.8539064697323621, 0.6217258472778029, 1.301443317879174, 0.6717846214753251, 0.6705916820858042, 1.0041619368720864, 0.9390000000000001, 1.0355688575802984, 0.6773, 0.6612284906944133, 0.9189958592266666, 1.47792575389212, 0.742181607665194, 0.6996180865330766, 0.7875, 0.7287724273733078, 0.7735, 0.7789987332339462, 0.6628516700898398, 0.7144517958901624, 0.8201, 0.8035640333176906, 0.6097147603728045, 0.8956623892599755, 0.6259019931522362, 0.7298, 0.9856999999999999, 0.9443, 0.9991, 0.6666122275597149, 0.6618533701362705, 1.8118790523676156, 1.7415062036137683, 0.6547005670645635, 0.7671678669592431, 1.4728186072078, 1.4294601267494214, 0.7249, 0.7461, 0.6837398970905949, 1.4828041606403457, 1.2298594133575118, 0.6566367211624318, 1.7223240062390697, 0.5948491986957343, 1.1012990724532667, 1.2129773577483747, 0.8750566129706747, 0.6865082357479647, 0.6718791256992164, 1.141819725658855, 0.708660400831887, 0.7507014276983182, 0.6502469914501021, 0.6622089721823552, 0.8744, 0.790195341375317, 0.7003462531139196, 0.8955, 0.8088466440259665, 0.9756527513957738, 1.0481290764284352, 0.7864759385609773, 0.877, 1.0239177007187794, 0.7367106602580592, 1.2112883462999684, 0.7893550356313082, 0.8877000373040718, 0.646591357107796, 0.7940999999999999, 1.4506637255655888, 0.7956404416159573, 0.828152291785582, 0.9738427759316091, 0.6197462935116648, 1.0, 0.5784582435668023, 1.0161926414006346, 0.6518, 0.9120185205639317, 0.5962331383079689, 0.8029460861652988, 0.9962999999999999, 0.6516584691374528, 0.8432, 0.7247680597116762, 0.8293232817655684, 0.7103703411435032, 0.6795, 0.593013099917073, 0.6870858254387268, 0.8987822258696679, 1.0113544885549315, 1.0623193147249348, 0.6384961659588667, 0.6673977518639199, 1.1735548975218888, 0.642882260005456, 0.9821152230495913, 0.6277806937918028, 1.5833781407044853, 0.7188440697151264, 0.916254268394175, 1.104117844637988, 0.7486046384933707, 0.6033688626290045, 0.6774, 0.6845676982522136, 0.7293084751466716, 1.5766595115722524, 0.8857383166201774, 0.9707000000000001, 0.7124167931418028, 1.0, 1.0240147313166492, 1.0, 1.469531569608193, 0.9081999999999999, 0.6605585674595652, 0.7343000000000001, 1.7827734540994187, 1.092979854205574, 0.9027, 0.6980548148072092, 0.766009383930277, 0.6178367424609797, 1.01251736081918, 0.7033029456852328, 0.8095546067296375, 0.6714000000000001, 0.754664930676406, 0.9682521146169096, 0.6817008736190165, 0.7573000000000001, 0.8163489714236163, 0.7988658991403818, 0.7948999999999999, 0.6219167986005623, 0.6726477026013924, 0.7496382700802454, 0.6927814923333667, 0.9847840080559341, 0.7427029433778315, 0.7295486022641013, 1.7663249881719356, 0.7838226618447436, 1.0, 0.654606983691644, 0.6482088270455348, 0.740700297135664, 1.1732398227945526, 0.7189922582825344, 1.0658725688987045, 0.6412284453760765, 0.9843000000000001, 0.9816959736511691, 0.722091631226171, 0.6849, 1.1272006044723966, 1.7388115497523606, 0.595937350678158, 0.4770816392780122, 0.7466295617598344, 1.4250437618941898, 0.795937471270875, 1.1950295562511948, 0.7126159468757342, 1.6226805047214525, 0.6424796177370243, 0.7115683182258155, 1.3615735473934514, 0.9837, 0.7780995539274157, 0.7226516453733989, 0.6912641491631796, 0.6670999999999999, 0.6994, 0.8590730988690518, 0.7567692157056005, 0.7696885911428412, 0.686955629064929, 0.7762045644229661, 0.7838998616897872, 0.8661914177527358, 0.7044398843160548, 0.6639522059103212, 1.0769415581995352, 0.9152318322655798, 0.7068066223673694, 0.777388348746955, 1.521155836589689, 0.7176971478800387, 0.7338961057264233, 0.708, 0.9340999999999999, 1.1602073021422272, 0.9163442094574835, 0.8802000000000001, 1.0454274382200959, 0.6547, 1.8339586464040776, 0.6833, 0.6944293488461801, 1.3226231968371367, 0.8436360110007622, 0.49218491621632837, 0.718487348986657, 1.8161744314159138, 0.7322729792367858, 0.3095674632886007, 0.8713772495126748, 0.5828539810449793, 0.5648950526290734, 0.7280757926367866, 1.0071361336367637, 0.9867, 1.0486784049294946, 0.7134315108333196, 0.7945, 0.0, 1.7360867491690912, 1.374692167595769, 1.0, 0.974796948674384, 0.6259726214873664, 0.6228494523350382, 0.8427035932681708, 1.2891256989508415, 1.2004050254595755, 0.7301971967430252, 0.6362073642215254, 0.8337263170496337, 1.3649086471921321, 0.7109000000000001, 1.3522052966869458, 0.6017956034133287, 0.6049182929749058, 0.8663406819105042, 0.8412308105807323, 0.8795000000000001, 0.6114281642453554, 1.259185875600395, 0.7911529813811453, 0.7636291949732866, 0.020871621060584798, 0.734847196017428, 0.6889, 0.5991632838376111, 0.6479812310111017, 1.9785125834425419, 1.6997991850864176, 0.9893, 0.8639867546308044, 0.7010961763737634, 0.8789518994642442, 0.7136567572009855, 1.9207085873802565, 0.6334962306265064, 0.6086583701085101, 0.5920856069870742, 1.6532975475771332, 0.9765587009153794, 0.6388829371865289, 0.6571693441594706, 1.110578703338175, 0.7354568391528011, 0.9966444624455829, 0.6803954388255246, 1.110917036210628, 0.6564695427380721, 0.6748999999999999, 0.7178, 0.7377438183707613, 0.6408997322003139, 0.8266, 0.6739468439379788, 0.7453410454779317, 0.851, 0.7251000000000001, 0.6388622789060259, 0.6652527888730149, 0.7671357801411699, 0.866778671139933, 0.7229218811087478, 1.0850400449560629, 0.7596808427417578, 0.7779, 0.6220058219824546, 0.9993000000000001, 0.6623408077325947, 0.733090263346265, 0.9239771307241031, 0.6918708500952211, 1.4719201373657638, 0.7020045874306227, 0.6474227146935366, 1.2455905865932495, 1.5625164907098834, 0.8328157576361469, 1.0, 0.8423959478728822, 1.0928326201531875, 0.9713651192205195, 0.8932202274369013, 0.6606000000000001, 0.824813469022787, 0.6258105710269544, 0.69833602033928, 0.6388666420412706, 0.69736602054219, 1.0, 0.9684546108990999, 0.5995, 1.4804830314079722, 1.278642974620405, 0.7988556679444727, 0.6267014828441089, 0.8946925809708167, 0.6156703950513271, 0.6457999999999999, 0.8600788391644865, 0.6739330869748078, 1.0318102351540908, 0.8327365302193116, 0.6812675468444698, 1.0, 0.8661239395035665, 0.982, 0.9791, 0.9993794437528454, 0.7408997006304234, 0.5644319242722117, 0.7700779446550653, 0.8212339883632335, 0.6513909230856397, 0.7394722880965839, 1.5828582630186194, 0.7240431956829039, 1.0041911615596415, 1.050302910917257, 0.7, 0.6655, 1.0077006553303531, 0.8289421560493262, 0.8516360811735161, 0.693661460224585, 0.7862, 0.6937872060349484, 0.8918293663385174, 0.6761302919150095, 0.8696505924507958, 1.7114251052958278]
const global poll_vote = [0.8932882665379044, 0.9805701834029417, 0.814071812763483, 1.0050251256281406, 0.9161859518154055, 0.7751937984496124, 1.2793176972281448, 0.9857612267250822, 0.7945850500294291, 1.0479041916167666, 1.0226691665246292, 0.8573818485989128, 0.999846177511152, 0.9459268671466893, 0.7155635062611807, 0.9888751545117429, 0.7980049875311721, 0.7726889575723517, 0.7824033067611457, 0.963792654337067, 1.1760966306420853, 0.9202453987730062, 0.7529280535415505, 1.0135135135135136, 0.9336890243902439, 1.0814249363867685, 1.0202550637659413, 1.0981584727149858, 0.8969248291571754, 0.9140201394268009, 1.0638297872340425, 0.7458048477315102, 0.9522322822354042, 0.8965929468021518, 0.8207273897650467, 0.8721804511278195, 0.8061420345489443, 1.0230654761904763, 0.8761508761508761, 0.77007700770077, 0.964840556009812, 0.7949561403508772, 1.0169491525423728, 0.5818701510820744, 0.7671326287078077, 0.8536199810306672, 0.8486562942008486, 1.1145623972227299, 0.7152211141331721, 0.9310682783404116, 0.47705002578648786, 0.7828639773839295, 1.072234762979684, 1.0896483407627537, 0.7779506557012669, 1.0720268006700167, 0.757201646090535, 0.777921837377292, 0.7922705314009661, 1.0261194029850746, 0.7838745800671892, 0.6944444444444444, 0.7722007722007722, 0.7993338884263115, 0.8604997517789178, 0.9048453006421483, 0.7581501137225171, 0.9032038173142468, 1.330292664386165, 1.0269953051643192, 1.0212222754108824, 0.8667822376316843, 0.7671232876712328, 0.9480222294867604, 1.0567048667956542, 0.8135869012508898, 0.8700980392156863, 0.7294738475865279, 0.9193892628468232, 0.8708708708708709, 1.0726474890297415, 0.732526197985553, 0.8705114254624592, 0.7498295841854125, 0.9296148738379814, 0.9603841536614646, 1.0240112994350283, 0.8418043202033036, 1.305379746835443, 1.0825439783491204, 0.8029548739360848, 0.8648381517173215, 1.081752083702784, 0.67, 0.8248730964467005, 1.0832559579077685, 0.8194554586307164, 1.831174631531934, 0.8843169103049991, 0.9765886287625418, 0.8262755629002273, 0.9582689335394127, 0.8219881839198562, 0.9497964721845318, 0.9166780681351757, 1.0165745856353592, 1.2160228898426324, 1.1823996898623763, 0.8933662801748717, 1.1070110701107012, 0.8585994097129057, 1.0223455527968452, 0.9675858732462506, 0.9035755776429586, 1.0904425914047466, 0.7834441980783443, 0.8807045636509206, 0.7515769695342907, 0.4448964026948011, 0.8462211847096586, 1.0024621878297573, 0.8689607229753216, 0.7795404814004376, 0.8294209702660407, 0.961032122169563, 1.163586584531143, 0.8577666874610106, 0.9492168960607499, 0.8239448461409113, 0.9022786358770454, 0.7387140902872777, 0.9020182658698839, 0.9693679720822025, 0.8062418725617685, 0.8888888888888888, 0.8211678832116788, 0.9071416790367804, 0.9254627313656829, 0.9138940454091103, 0.8650519031141869, 0.86687306501548, 1.0765550239234452, 0.9505080301540478, 0.8152591908937086, 0.8370436331255565, 0.9107956067506028, 0.7915213308290849, 0.8813250265917033, 1.0738752082947602, 0.8008153756552125, 0.852151682999574, 0.668364099299809, 0.878409616273694, 0.730631375792666, 1.1302575504910135, 0.9391007398975526, 0.5115712545676006, 0.8497409326424871, 0.6834268977300464, 0.6728778467908902, 0.9235936188077246, 0.9386192416944548, 1.3995334888370543, 1.0701373825018077, 0.8497932935231971, 0.9752020061298412, 0.7281048470979821, 0.9736236502035758, 0.888529886914378, 0.9391007398975526, 0.741809190191634, 0.7849109653233365, 0.7948947604120018, 0.9254864736592311, 1.1096316023080337, 0.9291023441966839, 0.7875647668393783, 0.9722897423432183, 0.8788159111933395, 0.7980845969672785, 1.0502625656414102, 0.9172909341079346, 0.64, 0.6325581395348837, 0.8017492711370263, 1.0010355540214015, 0.8657351154313487, 0.7964889466840052, 0.964274423016124, 0.7946805060006488, 1.0116842405243658, 0.7862033984275932, 0.8208594881699661, 1.0230179028132993, 0.8363201911589008, 1.0818356307799686, 0.8264462809917356, 0.8680169152014244, 0.8573878250928836, 0.7884813164209804, 0.4681647940074906, 0.9548611111111112, 0.862966820413629, 0.8048289738430584, 0.7432337680549712, 0.9379968203497615, 0.8969792903310908, 0.9240246406570841, 0.8855154965211891, 0.8900705120795284, 0.9822505600551439, 0.984060984060984, 0.6114918292040064, 0.912200684150513, 0.9164420485175202, 0.9099181073703366, 0.9784735812133072, 0.8433568367205863, 1.0668340131785379, 0.8592910848549946, 1.1052937754508434, 0.6900750964075502, 0.7847271023658937, 0.8977791778232792, 0.8460787504067686, 0.898842169408897, 0.8163265306122449, 0.9016134134767478, 1.0153541357107478, 0.8608058608058607, 0.9156438729787649, 1.0819949281487744, 1.1882998171846435, 0.9245942058763098, 0.9182530795072789, 0.9917704156995146, 0.9839461418953911, 0.9106933019976499, 0.9222886421861657, 0.9811208562509292, 0.8904719501335708, 0.9275618374558303, 0.983606557377049, 1.2152777777777777, 0.5770340450086555, 0.9389671361502347, 0.8137169427492009, 0.7258900795022468, 0.8058227190018197, 0.7948335817188276, 0.7686395080707148, 1.0045823052520269, 0.8808724832214765, 0.8744231236337139, 0.94976255936016, 1.1181192660550459, 0.9026434558349452, 0.7200720072007201, 0.870213823968175, 0.7638252367858235, 0.9539473684210527, 0.6496272630457933, 0.8268123431271224, 1.1410788381742738, 0.9151822193168819, 0.762970498474059, 1.0027347310847765, 0.9297520661157025, 0.8051082731815657, 0.8631319358816276, 0.9142857142857143, 0.7725321888412017, 0.8568708346556648, 0.9459268671466893, 0.9179056237879768, 0.8522136769902308, 1.0091600683123738, 0.9658978907944018, 0.7925862699670772, 0.8686765457332651, 0.9133178346064431, 0.6402655916528337, 0.9327758121582502, 0.84773394196283, 0.9239363987967341, 0.9728692792545902, 0.9605829744948658, 1.1350407450523865, 0.9588268471517202, 0.649284772243076, 0.8600237247924081, 0.7095202795721698, 1.024279210925645, 0.8607746972275048, 0.8800521512385919, 0.8890251379521766, 1.190771520714463, 0.9137508357477157, 1.0924369747899159, 0.7833603457590491, 1.0618023414102915, 0.8035251425609125, 1.2520413718018508, 1.002889682134965, 1.0185427004439802, 0.7214428857715431, 0.7421594445774479, 0.9968282736746715, 0.9604829857299672, 0.9821038847664775, 0.9721156305960603, 0.8451957295373665, 0.9242654159194372, 0.8041817450743868, 0.7692307692307693, 0.7077140835102618, 1.14140773620799, 1.1568123393316194, 0.9321175278622087, 0.9077878643096035, 0.8433518751121479, 1.229050279329609, 0.9085636961144404, 0.9015256588072122, 0.9924262209454165, 0.7636577246915998, 1.054481546572935, 0.400962309542903, 1.034540297013182, 0.8294374905745739, 1.028695181375203, 0.9784075573549258, 0.9861139062185551, 0.9487237406821775, 0.8673186934858831, 0.9236826165960024, 0.8577310155535225, 1.770293609671848, 0.9198423127463864, 0.6253489670575098, 0.9409474367293965, 0.9032943676939426, 0.8468052347959969, 0.9673748103186647, 0.8816457387122629, 0.9043927648578811, 0.9483667017913593, 0.935005701254276, 0.8933002481389579, 0.9587727708533078, 1.014304291287386, 0.9119496855345912, 0.9975816203143895, 1.1670020120724345, 0.46425255338904364, 0.7156549520766773, 0.8620689655172413, 0.9719710669077757, 1.0151491488364828, 0.7681652184863368, 1.069055186362323, 1.0995723885155773, 0.9137328675087342, 0.9199274423425757, 0.8996345234748383, 0.9543836945972177, 1.149425287356322, 0.92, 1.0293748430831033, 0.8752327746741154, 0.9667376699983614, 0.9665541577170911, 0.955819881053526, 0.8985200845665963, 0.7807981492192019, 0.9811046511627907, 0.8056719303899452, 0.4617083207869116, 1.0986929342678537, 0.7827324478178369, 1.038637307852098, 0.8149713250830063, 0.9595327492699208, 1.142032029404043, 0.8769667268506577, 0.5436820397450318, 0.8899297423887588, 0.8977189109639441, 1.192368839427663, 0.7164179104477612, 0.8828996282527881, 0.9237187127532777, 0.9154155986818016, 0.7390202702702703, 0.7434944237918215, 1.0065288356909685, 1.0509296685529506, 0.8763837638376383, 1.2568735271013354, 0.7797270955165693, 1.013724266999376, 1.0457125784284433, 0.8349618049387103, 0.8612284692882678, 0.9677419354838709, 0.9146341463414634, 0.8848252470137148, 0.7932573128408527, 0.7858546168958742, 0.8302583025830258, 0.6505387273836146, 1.1386811692726035, 0.8639910813823857, 0.6215173596228033, 0.9152642456451138, 0.8956100425781823, 0.8018953891015127, 1.2195121951219512, 1.0976639459611595, 0.8447512104666735, 1.0276280323450135, 0.71, 0.9151223976206817, 0.72, 0.7635302043566135, 0.6606474344857961, 1.062215477996965, 0.5855917200054473, 1.246591351772497, 1.0630023334197563, 0.8505209440782479, 0.8308408109006314, 0.7121336620104082, 1.2360446570972887, 0.9805472086035111, 0.8493771234428086, 0.9230332291962511, 0.8179339594062406, 0.9446693657219973, 0.7745010425975574, 0.9598080383923214, 0.8623012665049852, 0.896057347670251, 0.867177337765573, 1.1344952318316344, 0.804082263800835, 0.7922883929750428, 0.8798240351929615, 0.9450945094509452, 0.9812555038369607, 1.0126132527980103, 0.9352787130564908, 0.8445945945945945, 0.8398320335932813, 0.9219668626402993, 0.7905138339920948, 1.0377527285739847, 1.0912962470055896, 1.0164835164835164, 0.9658502932045534, 0.704083685375176, 0.7904061052057781, 0.7439712673165727, 0.8, 1.0274497776414662, 0.6418769366976539, 0.9577464788732394, 1.2593601089176312, 1.068566340160285, 0.7439712673165727, 0.8035172832019406, 0.38602194229987813, 0.9649122807017543, 1.0727455581629233, 0.8201523140011716, 0.8518161362905818, 0.6705272782688204, 0.8854781582054309, 0.8856905618599501, 0.766571471516609, 0.8468389545919113, 0.8236239453595822, 0.8810572687224669, 0.8796481407437025, 0.9945908218461001, 0.6426276329882185, 0.8657373440939105, 0.7261653783087374, 0.40537116797567774, 0.8222153273347812, 0.8835196538045438, 0.969797727902466, 0.9224219489120151, 0.8071367884451996, 1.0146003464488988, 1.086272040302267, 0.9417808219178082, 0.9325114588272483, 0.6142506142506142, 0.9945130315500685, 1.0599721059972105, 0.6506048592050422, 1.0902483343428224, 0.764168966249204, 0.8393461934913857, 1.054713249835201, 0.8544446110028482, 1.1189634864546525, 0.929368029739777, 0.8086253369272237, 0.9123013537374928, 0.8518225039619652, 0.8308157099697886, 0.8647798742138365, 0.864131879819197, 1.0535117056856187, 0.7074212493326214, 0.800711743772242, 0.8168028004667445, 0.6987916727325666, 0.8939455505891915, 0.998185117967332, 0.9726443768996961, 0.8074382187423539, 0.9174311926605504, 0.9238998298079261, 0.916774101131643, 1.1830819284235432, 0.6894232386077014, 0.8586639189421261, 0.8333333333333334, 0.903954802259887, 1.0207237859573153, 0.6530350069585698, 1.021021021021021, 0.6297972839992128, 0.7433102081268583, 0.568052715291979, 0.968872397443826, 0.9164502825721704, 1.0084033613445378, 0.9073613346992536, 0.9636443276390715, 1.0574712643678161, 1.2898024989923418, 0.8805031446540881, 0.8208955223880596, 0.9770266701874836, 0.9610250934329951, 1.0161542470036478, 0.849532037437005, 0.9830508474576272, 0.8585503166783955, 0.7905900501349789, 0.9244644870349492, 0.8582695430294596, 1.0894802643329164, 1.0287485907553553, 0.8168451624614267, 0.7575757575757576, 0.7803790412486065, 0.9137426900584795, 0.8980355472404116, 0.9220985691573927, 0.7882636303919421, 0.7300188797986155, 0.8366141732283464, 0.8741258741258742, 0.795196364816618, 0.72, 0.9034712315739419, 1.3258541560428354, 0.8665511265164645, 0.8595284872298625, 1.2015018773466832, 1.0682004930156122, 0.9633121541299446, 0.9010812975570686, 0.955137481910275, 0.8945288121489494, 0.8272058823529412, 0.9629935341862705, 0.9587260318492038, 0.9173174662204041, 0.8285163776493256, 0.9424672949781966, 0.8508992457938503, 0.9984768996446098, 1.208617971623752, 0.8404773911581778, 0.9910802775024777, 0.9237875288683604, 0.8091517857142856, 0.6253553155201819, 0.973096737263881, 0.9548896937767534, 0.6227758007117438, 0.7751937984496124, 0.7360813704496788, 1.1286681715575622, 0.9351927809680065, 0.9019426456984273, 0.8419219044854115, 0.9324814367121396, 1.015375689004932, 0.9593859929645028, 0.9470934030045721, 0.7657247037374659, 0.6772465379561307, 0.9050823857043396, 1.07671601615074, 0.8915304606240714, 0.9439359267734554, 0.8935039845447961, 1.6812865497076024, 0.9449881876476545, 1.0604870384917517, 1.1154011154011154, 0.7987711213517665, 0.9533201840894148, 0.9043635541487679, 0.9855564995751911, 0.9789029535864979, 0.8747105737072293, 0.8310692024932076, 0.9033423667570009, 0.857796416317194, 0.9833916083916084, 0.7556675062972292, 0.7253232418795332, 0.877681805516857, 1.0040640688501077, 0.8596783139212423, 1.0279605263157894, 1.079707843759924, 0.7705692269450659, 0.6290829905637552, 0.9760065067100447, 0.9432527004411988, 0.9933774834437086, 0.9194289862085653, 0.797816502204493, 0.8813160987074031, 0.7447248655357881, 0.6401766004415012, 0.9098567818028643, 1.1225144323284157, 0.8604564160119715, 0.843633509907789, 0.8916990920881972, 1.0453453877666337, 0.8438818565400844, 0.8636017272034545, 0.8504606661941886, 0.9484193011647254, 0.8870034708831469, 0.9054086252084823, 1.193633952254642, 0.7505253677574302, 0.8306414397784956, 0.8366291451171282, 0.425531914893617, 0.7666867107636801, 0.8373509261608729, 0.6419201786212672, 0.8559732043170822, 0.8426120975022571, 0.9033121445299431, 0.8792302587923025, 0.7826887661141805, 0.8783487044356608, 0.9555009555009555, 0.67, 0.8987126548457615, 0.5187618882932734, 0.7184672698243747, 0.9369951534733442, 1.022176022176022, 0.9688162276718135, 1.0403329065300897, 0.9148486980999295, 1.1512134411947728, 1.1660888748818152, 0.9945460378569136, 0.7886953664147224, 0.74, 0.8816547982366905, 0.7839866555462885, 0.7492975335622853, 0.9305417082087072, 0.9133307423241352, 0.9829280910501812, 1.0742420625447602, 0.9306963603124481, 0.9628345850182939, 0.9907874152616026, 0.9426751592356688, 0.9600495509445649, 1.0533245556287032, 0.9471939379587971, 0.8822470291681671, 0.997697620874904, 0.8822401227464518, 0.8357915437561455, 0.59, 0.9880749574105622, 0.8565310492505354, 0.5622097286726963, 0.6517311608961304, 1.0407632263660016, 0.6536615258911245, 1.0003077870113881, 0.897460378079053, 0.8707865168539326, 0.7822448608331818, 0.9559050262102992, 0.7055961070559611, 0.8850956219377272, 1.058761249338274, 0.8485519276886183, 0.945766218412886, 1.0373443983402488, 0.8078335373317013, 1.007838745800672, 0.8661417322834646, 0.9142857142857143, 0.9767092411720512, 0.932806324110672, 0.4219409282700422, 0.9509875640087784, 0.9516041326808048, 0.9782608695652175, 1.0253781081773905, 0.7958615200955034, 1.0932775063968365, 0.8946105171285185, 0.8522004578987534, 1.071055381400209, 0.9640666082383873, 0.8985879332477535, 0.9264978381717108, 0.8090614886731392]
