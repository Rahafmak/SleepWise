import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# SLEEPWISE AI — FRIENDLY SLEEP-QUALITY CLASSIFIER
# ============================================================

st.set_page_config(
    page_title="SleepWise AI",
    page_icon="🌙",
    layout="wide"
)

# Always load files relative to app.py.
# This prevents FileNotFoundError when Streamlit is launched
# from a different working directory.
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "sleepwise_model.pkl"
FEATURES_PATH = BASE_DIR / "sleepwise_features.pkl"
IMPORTANCE_PATH = BASE_DIR / "sleepwise_feature_importance.pkl"
METRICS_PATH = BASE_DIR / "sleepwise_metrics.pkl"
CLEAN_DATA_PATH = BASE_DIR / "sleepwise_clean_data.csv"
COMPARISON_PATH = BASE_DIR / "sleepwise_model_comparison.csv"


# ------------------------------------------------------------
# LOAD DEPLOYMENT ARTIFACTS
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_pickle(path, default=None):
    if path.exists():
        return joblib.load(path)
    return default


@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# The model is required.
if not MODEL_PATH.exists():
    st.error(
        "⚠️ The trained model file was not found.\n\n"
        f"Expected location:\n`{MODEL_PATH}`\n\n"
        "Please run the final deployment/save cell in the notebook first. "
        "It will create the required `.pkl` and `.csv` files."
    )
    st.stop()

try:
    model = load_model()
    features = load_pickle(FEATURES_PATH, default=[])
    importance_df = load_pickle(
        IMPORTANCE_PATH,
        default=pd.DataFrame()
    )
    metrics = load_pickle(
        METRICS_PATH,
        default={}
    )
    clean_data = load_csv(CLEAN_DATA_PATH)
    comparison_df = load_csv(COMPARISON_PATH)

except Exception as e:
    st.error(
        "I couldn't load the SleepWise deployment files. "
        "Make sure you ran the final notebook save cell and that all "
        "generated files are in the same folder as app.py."
    )
    st.exception(e)
    st.stop()


# ------------------------------------------------------------
# USER-FACING OPTIONS
# These must match the categories used during model training.
# ------------------------------------------------------------

SLEEP_DURATION_OPTIONS = [
    "0–2 hours",
    "3–5 hours",
    "6–8 hours",
    "9–11+ hours"
]

# The trained notebook currently uses Sleep_Duration_Hours
# as an engineered numeric feature.
SLEEP_DURATION_MAP = {
    "0–2 hours": 1,
    "3–5 hours": 2,
    "6–8 hours": 3,
    "9–11+ hours": 4
}

SCREEN_TIME_OPTIONS = [
    "0–1 hours",
    "1–2 hours",
    "2–3 hours",
    "3+ hours"
]

CAFFEINE_TIMING_OPTIONS = [
    "Never",
    "6+ hours before bed",
    "3–5 hours before bed",
    "0–2 hours before bed"
]

EXERCISE_OPTIONS = [
    "No Exercise",
    "Under 30 Minutes",
    "30-60 Minutes",
    "1-2 Hours",
    "2+"
]

CLASS_STYLE = {
    "Good": (
        "🟢",
        "Nice — your profile is associated with good predicted sleep quality."
    ),
    "Fair": (
        "🟡",
        "Your profile lands in the fair zone — there may be room to improve."
    ),
    "Poor": (
        "🔴",
        "Your profile is associated with poor predicted sleep quality."
    ),
}


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def prepare_user_profile(user_input):
    """
    Accept a friendly user profile and convert it into the exact
    feature format expected by the trained pipeline.

    This supports the same format used in the notebook example:
    'Sleep Duration': '3–5 hours'
    """
    profile = dict(user_input)

    if (
        "Sleep Duration" in profile
        and "Sleep_Duration_Hours" not in profile
    ):
        profile["Sleep_Duration_Hours"] = SLEEP_DURATION_MAP[
            profile["Sleep Duration"]
        ]
        profile.pop("Sleep Duration", None)

    # Keep only features used by the trained model.
    profile = {
        key: value
        for key, value in profile.items()
        if key in features
    }

    return pd.DataFrame([profile])


def predict_sleep_quality(
    user_input,
    pipeline=model
):
    """
    Same prediction logic used by the notebook's
    'Try It Out on One Person' section.
    """
    input_df = prepare_user_profile(user_input)

    prediction = pipeline.predict(input_df)[0]

    probabilities = dict(
        zip(
            pipeline.classes_,
            pipeline.predict_proba(input_df)[0]
        )
    )

    return prediction, probabilities


def compare_what_if(
    current_profile,
    changes,
    pipeline=model
):
    """
    Same What-If logic used in the notebook:
    compare the current profile against a modified scenario.
    """
    scenario_profile = {
        **current_profile,
        **changes
    }

    current_pred, current_probs = predict_sleep_quality(
        current_profile,
        pipeline
    )

    scenario_pred, scenario_probs = predict_sleep_quality(
        scenario_profile,
        pipeline
    )

    return {
        "current_prediction": current_pred,
        "scenario_prediction": scenario_pred,
        "current_probabilities": current_probs,
        "scenario_probabilities": scenario_probs,
    }


def show_result_card(
    prediction,
    probabilities,
    heading="Your Sleep Insights"
):
    icon, message = CLASS_STYLE.get(
        prediction,
        ("⚪", "")
    )

    st.header(
        f"🔍 {heading}"
    )

    banner = {
        "Good": st.success,
        "Fair": st.warning,
        "Poor": st.error
    }.get(
        prediction,
        st.info
    )

    banner(
        f"{icon} Predicted Sleep Quality: **{prediction}**"
    )

    st.write(message)

    st.subheader(
        "How confident is the model?"
    )

    cols = st.columns(3)

    for col, label in zip(
        cols,
        ["Poor", "Fair", "Good"]
    ):

        p = probabilities.get(
            label,
            0
        )

        with col:
            st.metric(
                label,
                f"{p:.0%}"
            )

            st.progress(
                float(p)
            )


# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.sidebar.title("🌙 SleepWise AI")

page = st.sidebar.radio(
    "Navigate",
    [
        "Sleep Check-In",
        "Project Overview"
    ]
)


# ============================================================
# PAGE 1 — SLEEP CHECK-IN
# ============================================================

if page == "Sleep Check-In":

    st.title("🌙 SleepWise AI")

    st.markdown(
        """
        ### Predict. Understand. Explore.

        Answer a few questions about your sleep, lifestyle, and wellbeing.
        SleepWise AI will classify your predicted sleep quality as:

        🔴 **Poor** · 🟡 **Fair** · 🟢 **Good**

        Then explore the **What-If Lab** to see how the model responds
        when you change one or more habits.
        """
    )

    st.info(
        "This is an educational machine-learning project. "
        "Predictions are patterns learned from survey data, not a diagnosis "
        "or medical advice."
    )

    st.divider()

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    st.header(
        "🛌 1. Tell Me About Your Sleep"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=10,
            max_value=80,
            value=23
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    with col3:
        sleep_time = st.selectbox(
            "When do you usually fall asleep?",
            ["Night", "Morning"]
        )

    sleep_duration = st.select_slider(
        "How many hours do you usually sleep?",
        options=SLEEP_DURATION_OPTIONS,
        value="6–8 hours"
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    st.header(
        "📱 2. Your Evening Habits"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        screen_time = st.select_slider(
            "Screen time before bed",
            options=SCREEN_TIME_OPTIONS,
            value="1–2 hours"
        )

    with col2:
        caffeine_cups = st.slider(
            "Caffeinated drinks per day",
            0,
            4,
            value=2
        )

    with col3:
        caffeine_timing = st.selectbox(
            "When was your last caffeine before bed?",
            CAFFEINE_TIMING_OPTIONS,
            index=1
        )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    st.header(
        "🧠 3. Wellbeing & Lifestyle"
    )

    col1, col2 = st.columns(2)

    with col1:
        stress = st.slider(
            "Stress level",
            1,
            5,
            value=3,
            help="1 = very low stress, 5 = very high stress"
        )

    with col2:
        exercise = st.selectbox(
            "Daily exercise",
            EXERCISE_OPTIONS,
            index=1
        )

    st.divider()

    # --------------------------------------------------------
    # BUILD PROFILE
    # --------------------------------------------------------

    current_profile = {
        "Age": age,
        "Gender": gender,
        "Sleep Duration": sleep_duration,
        "Sleep Time": sleep_time,
        "Screen Time Before Bed": screen_time,
        "Caffeine Cups": caffeine_cups,
        "Caffeine Timing": caffeine_timing,
        "Exercise": exercise,
        "Stress Level": stress,
    }

    if st.button(
        "🌙 Predict My Sleep Quality",
        use_container_width=True,
        type="primary"
    ):

        prediction, probabilities = predict_sleep_quality(
            current_profile
        )

        st.session_state[
            "current_profile"
        ] = current_profile

        st.session_state[
            "current_prediction"
        ] = prediction

        st.session_state[
            "current_probabilities"
        ] = probabilities

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if "current_prediction" in st.session_state:

        prediction = st.session_state[
            "current_prediction"
        ]

        probabilities = st.session_state[
            "current_probabilities"
        ]

        show_result_card(
            prediction,
            probabilities
        )

        # ----------------------------------------------------
        # FACTOR IMPORTANCE
        # ----------------------------------------------------

        st.subheader(
            "🔍 What's influencing this prediction?"
        )

        st.caption(
            "These scores come from the model's behavior on the held-out "
            "test data. They describe predictive importance, not proof of "
            "medical or causal relationships."
        )

        if not importance_df.empty:

            for _, row in importance_df.head(5).iterrows():

                name = row["Feature"]
                importance = row["Importance Mean"]

                if importance > 0:

                    st.write(
                        f"**{name}** — "
                        f"importance score: {importance:.3f}"
                    )

                else:

                    st.write(
                        f"**{name}** — "
                        "low or uncertain importance in testing"
                    )

        # ----------------------------------------------------
        # WHAT-IF LAB
        # ----------------------------------------------------

        st.divider()

        st.header(
            "🔮 What-If Lab"
        )

        st.markdown(
            """
            Curious what might move the needle?

            Change one or more habits and compare the model's prediction
            with your current profile.
            """
        )

        whatif_col1, whatif_col2 = st.columns(2)

        with whatif_col1:

            whatif_stress = st.slider(
                "What if your stress level changed?",
                1,
                5,
                value=stress,
                key="whatif_stress"
            )

            whatif_screen = st.select_slider(
                "What if your screen time changed?",
                options=SCREEN_TIME_OPTIONS,
                value=screen_time,
                key="whatif_screen"
            )

            whatif_duration = st.select_slider(
                "What if your sleep duration changed?",
                options=SLEEP_DURATION_OPTIONS,
                value=sleep_duration,
                key="whatif_duration"
            )

        with whatif_col2:

            whatif_caffeine = st.selectbox(
                "What if your caffeine timing changed?",
                CAFFEINE_TIMING_OPTIONS,
                index=CAFFEINE_TIMING_OPTIONS.index(
                    caffeine_timing
                ),
                key="whatif_caffeine"
            )

            whatif_exercise = st.selectbox(
                "What if your exercise changed?",
                EXERCISE_OPTIONS,
                index=EXERCISE_OPTIONS.index(
                    exercise
                ),
                key="whatif_exercise"
            )

        if st.button(
            "✨ Explore This Scenario",
            use_container_width=True
        ):

            changes = {
                "Stress Level": whatif_stress,
                "Screen Time Before Bed": whatif_screen,
                "Sleep Duration": whatif_duration,
                "Caffeine Timing": whatif_caffeine,
                "Exercise": whatif_exercise,
            }

            scenario = compare_what_if(
                st.session_state[
                    "current_profile"
                ],
                changes
            )

            st.subheader(
                "🔄 Scenario Comparison"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### Current Profile"
                )

                icon, _ = CLASS_STYLE.get(
                    scenario[
                        "current_prediction"
                    ],
                    ("⚪", "")
                )

                st.metric(
                    "Prediction",
                    f"{icon} "
                    f"{scenario['current_prediction']}"
                )

                for label in [
                    "Poor",
                    "Fair",
                    "Good"
                ]:

                    st.write(
                        f"{label}: "
                        f"{scenario['current_probabilities'].get(label, 0):.1%}"
                    )

            with col2:

                st.markdown(
                    "### What-If Scenario"
                )

                icon, _ = CLASS_STYLE.get(
                    scenario[
                        "scenario_prediction"
                    ],
                    ("⚪", "")
                )

                st.metric(
                    "Prediction",
                    f"{icon} "
                    f"{scenario['scenario_prediction']}"
                )

                for label in [
                    "Poor",
                    "Fair",
                    "Good"
                ]:

                    st.write(
                        f"{label}: "
                        f"{scenario['scenario_probabilities'].get(label, 0):.1%}"
                    )

            if (
                scenario["current_prediction"]
                != scenario["scenario_prediction"]
            ):

                st.success(
                    "Under this scenario, the model's prediction shifts "
                    f"from **{scenario['current_prediction']}** to "
                    f"**{scenario['scenario_prediction']}**."
                )

            else:

                st.info(
                    "The predicted class stays "
                    f"**{scenario['scenario_prediction']}**, although "
                    "the underlying probabilities may have changed."
                )

            st.warning(
                "This What-If result is a model-based simulation. "
                "It does not guarantee that changing these habits will "
                "change real-world sleep quality."
            )


# ============================================================
# PAGE 2 — PROJECT OVERVIEW
# ============================================================

else:

    st.title(
        "📊 SleepWise AI — Project Overview"
    )

    st.markdown(
        """
        This page presents the machine-learning project from problem
        definition and exploratory analysis through model comparison,
        evaluation, and deployment.
        """
    )

    # --------------------------------------------------------
    # PROBLEM DEFINITION
    # --------------------------------------------------------

    st.header(
        "🎯 Problem Definition"
    )

    st.write(
        """
        Sleep quality can be associated with multiple lifestyle and
        wellbeing factors. SleepWise AI uses survey information to classify
        sleep quality into three understandable categories:
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error(
            "🔴 Poor\n\nSleep Quality Score: 1–2"
        )

    with col2:
        st.warning(
            "🟡 Fair\n\nSleep Quality Score: 3"
        )

    with col3:
        st.success(
            "🟢 Good\n\nSleep Quality Score: 4–5"
        )

    st.write(
        """
        The goal is not to diagnose sleep disorders. The goal is to
        demonstrate how a classification model can identify patterns
        in sleep, lifestyle, and wellbeing survey data and provide an
        interactive, explainable user experience.
        """
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.header(
        "📁 Dataset"
    )

    if not clean_data.empty:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Cleaned Rows",
                len(clean_data)
            )

        with col2:
            st.metric(
                "Features",
                len(features)
            )

        with col3:

            if "Age" in clean_data.columns:

                st.metric(
                    "Age Range",
                    f"{int(clean_data['Age'].min())}–"
                    f"{int(clean_data['Age'].max())}"
                )

        st.dataframe(
            clean_data.head(10),
            use_container_width=True
        )

    elif metrics:

        st.write(
            f"Training rows: "
            f"{metrics.get('n_training_rows', 'N/A')}"
        )

        st.write(
            f"Test rows: "
            f"{metrics.get('n_test_rows', 'N/A')}"
        )

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    st.header(
        "🔎 Exploratory Data Analysis"
    )

    if not clean_data.empty:

        if "Sleep Quality" in clean_data.columns:

            col1, col2 = st.columns(2)

            with col1:

                counts = (
                    clean_data[
                        "Sleep Quality"
                    ]
                    .value_counts()
                    .reindex(
                        [
                            "Poor",
                            "Fair",
                            "Good"
                        ]
                    )
                    .fillna(0)
                )

                st.bar_chart(
                    counts
                )

                st.caption(
                    "Distribution of Poor, Fair, and Good sleep-quality classes."
                )

            with col2:

                numeric_cols = [
                    col for col in [
                        "Age",
                        "Sleep_Duration_Hours",
                        "Caffeine Cups",
                        "Stress Level",
                        "Sleep Quality Score"
                    ]
                    if col in clean_data.columns
                ]

                if len(numeric_cols) >= 2:

                    corr = clean_data[
                        numeric_cols
                    ].corr()

                    st.dataframe(
                        corr.round(2),
                        use_container_width=True
                    )

                    st.caption(
                        "Correlation matrix for available numerical variables."
                    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.header(
        "🤖 Model Comparison"
    )

    if not comparison_df.empty:

        st.dataframe(
            comparison_df.round(3),
            use_container_width=True
        )

        if "CV Macro F1" in comparison_df.columns:

            chart_df = (
                comparison_df
                .set_index("Model")[
                    ["CV Macro F1"]
                ]
            )

            st.bar_chart(
                chart_df
            )

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    st.header(
        "🏆 Final Model Performance"
    )

    if metrics:

        model_name = metrics.get(
            "best_model_name",
            "Unknown"
        )

        cv_f1 = metrics.get(
            "cv_macro_f1"
        )

        test_acc = metrics.get(
            "test_accuracy"
        )

        test_precision = metrics.get(
            "test_precision"
        )

        test_recall = metrics.get(
            "test_recall"
        )

        test_f1 = metrics.get(
            "test_f1"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Test Accuracy",
                f"{test_acc:.1%}"
                if test_acc is not None
                else "N/A"
            )

        with col2:
            st.metric(
                "Macro Precision",
                f"{test_precision:.2f}"
                if test_precision is not None
                else "N/A"
            )

        with col3:
            st.metric(
                "Macro Recall",
                f"{test_recall:.2f}"
                if test_recall is not None
                else "N/A"
            )

        with col4:
            st.metric(
                "Test Macro F1",
                f"{test_f1:.2f}"
                if test_f1 is not None
                else "N/A"
            )

        st.success(
            f"🏆 Selected model: **{model_name}**"
        )

        if cv_f1 is not None:

            st.write(
                f"The selected model achieved a 5-fold cross-validated "
                f"Macro F1 of **{cv_f1:.2f}** on the training data."
            )

        st.write(
            "Macro F1 was used as the primary model-selection metric "
            "because it treats the Poor, Fair, and Good classes equally."
        )

        if metrics.get("was_tuned"):

            st.info(
                "The selected model was further fine-tuned using grid search."
            )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    st.header(
        "⚙️ Preprocessing"
    )

    st.write(
        """
        The final deployment pipeline keeps preprocessing and modeling
        together so that the Streamlit app applies the same transformations
        used during training.

        - Numerical features → median imputation + scaling
        - Nominal categories such as Gender and Sleep Time →
          most-frequent imputation + One-Hot Encoding
        - Ordered categories such as Screen Time, Caffeine Timing,
          and Exercise → most-frequent imputation + Ordinal Encoding
        """
    )

    # --------------------------------------------------------
    # DEPLOYMENT + LIMITATIONS
    # --------------------------------------------------------

    st.header(
        "🚀 Deployment & Limitations"
    )

    st.write(
        """
        SleepWise AI is deployed as an interactive Streamlit application.
        The user can:

        1. Enter a sleep and lifestyle profile.
        2. Receive a Poor/Fair/Good prediction.
        3. View class probabilities.
        4. Explore model-based feature importance.
        5. Run What-If scenarios.
        6. Review the project methodology and model performance.
        """
    )

    st.warning(
        "The dataset is a survey sample and the model identifies "
        "patterns rather than proving causation. The What-If Simulator "
        "is therefore a model-based scenario tool, not a medical "
        "recommendation system."
    )

    if metrics:

        st.caption(
            "Final balancing approach reported by the training notebook: "
            f"{metrics.get('balancing_method', 'Not specified')}."
        )
