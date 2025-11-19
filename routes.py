import os
import json
from flask import (
    session, render_template, request, redirect,
    url_for, flash, send_file
)
from werkzeug.utils import secure_filename
from app import app, db
from flask_login import current_user, login_required
from models import Dataset
import pandas as pd
from datetime import datetime

# Updated Quality Functions
from quality import (
    detect_missing,
    detect_duplicates,
    detect_outliers,
    detect_label_issues,
    extract_ydata_overview_stats,
    auto_detect_target_column,
)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


# ===================================================
# HELPERS
# ===================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_df(filepath: str, original_name: str) -> pd.DataFrame:
    ext = original_name.lower().rsplit('.', 1)[-1]

    if ext in ("xlsx", "xls"):
        return pd.read_excel(filepath)

    if ext == "csv":
        try:
            return pd.read_csv(filepath, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(filepath, encoding="latin1")

    raise ValueError(f"Unsupported file extension: {ext}")


@app.before_request
def make_session_permanent():
    session.permanent = True


# ===================================================
# HOME / DASHBOARD
# ===================================================
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('auth.login'))


@app.route('/dashboard')
@login_required
def dashboard():
    datasets = (
        Dataset.query
        .filter_by(user_id=current_user.id)
        .order_by(Dataset.uploaded_at.desc())
        .all()
    )
    return render_template('dashboard.html', user=current_user, datasets=datasets)


# ===================================================
# FILE UPLOAD
# ===================================================
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == "":
            flash('No file selected.', 'error')
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_name = f"{current_user.id}_{timestamp}_{filename}"

            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_name)
            file.save(filepath)

            dataset = Dataset(
                user_id=current_user.id,
                filename=final_name,
                original_filename=filename,
                file_size=os.path.getsize(filepath),
            )

            db.session.add(dataset)
            db.session.commit()

            flash('File uploaded successfully!', 'success')
            return redirect(url_for('profile_dataset', dataset_id=dataset.id))

        flash('Invalid file type. Upload CSV or Excel.', 'error')
        return redirect(request.url)

    return render_template('upload.html', user=current_user)


# ===================================================
# PROFILE PAGE
# ===================================================
@app.route('/profile/<int:dataset_id>')
@login_required
def profile_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)

    if dataset.user_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('dashboard'))

    if not dataset.profile_generated:
        try:
            from ydata_profiling import ProfileReport

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], dataset.filename)
            df = load_df(filepath, dataset.original_filename)

            profile = ProfileReport(
                df,
                title=f"Profile - {dataset.original_filename}",
                explorative=True
            )

            profile_filename = f"profile_{dataset.id}.html"
            profile_path = os.path.join(app.config["UPLOAD_FOLDER"], profile_filename)
            profile.to_file(profile_path)

            dataset.profile_generated = True
            dataset.profile_path = profile_filename
            db.session.commit()

            flash("Profile report generated!", "success")

        except Exception as e:
            flash(f"Error generating profile: {e}", "error")
            return redirect(url_for("dashboard"))

    return render_template("profile.html", dataset=dataset, user=current_user)


# ===================================================
# VIEW PROFILE REPORT
# ===================================================
@app.route('/view_profile/<int:dataset_id>')
@login_required
def view_profile(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)

    if dataset.user_id != current_user.id:
        flash("Access denied.", "error")
        return redirect(url_for("dashboard"))

    if not dataset.profile_generated:
        return redirect(url_for("profile_dataset", dataset_id=dataset_id))

    return render_template("view_report.html", dataset=dataset, user=current_user)


@app.route('/profile_report/<int:dataset_id>')
@login_required
def profile_report(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)

    if dataset.user_id != current_user.id:
        return "Access denied", 403

    if not dataset.profile_generated:
        return "Profile not generated", 404

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], dataset.profile_path)
    return send_file(filepath)


# ===================================================
# DELETE DATASET
# ===================================================
@app.route('/delete/<int:dataset_id>', methods=['POST'])
@login_required
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)

    if dataset.user_id != current_user.id:
        flash("Access denied.", "error")
        return redirect(url_for('dashboard'))

    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset.filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        if dataset.profile_path:
            profile_file = os.path.join(app.config["UPLOAD_FOLDER"], dataset.profile_path)
            if os.path.exists(profile_file):
                os.remove(profile_file)

        db.session.delete(dataset)
        db.session.commit()
        flash("Dataset deleted.", "success")

    except Exception as e:
        flash(f"Error deleting dataset: {e}", "error")

    return redirect(url_for("dashboard"))


# ===================================================
# CHANGE TARGET COLUMN
# ===================================================
@app.route("/change_target_column/<int:dataset_id>", methods=["POST"])
@login_required
def change_target_column(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)

    if dataset.user_id != current_user.id:
        flash("Access denied.", "error")
        return redirect(url_for("dashboard"))

    new_target = request.form.get("target_col")
    if not new_target:
        flash("No target column selected.", "error")
        return redirect(url_for("quality_issues", dataset_id=dataset.id))

    session_key = f"manual_target_column_{dataset.id}"
    session[session_key] = new_target

    flash(f"Target column changed to: {new_target}", "success")
    return redirect(url_for("quality_issues", dataset_id=dataset.id))


# ===================================================
# QUALITY REPORT
# ===================================================
@app.route("/quality/<int:dataset_id>")
@login_required
def quality_issues(dataset_id):

    dataset = Dataset.query.get_or_404(dataset_id)
    dataset_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset.filename)
    df = load_df(dataset_path, dataset.original_filename)

    # ===================================================
    # YDATA SUMMARY
    # ===================================================
    y_stats = None
    if dataset.profile_generated and dataset.profile_path:
        html_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset.profile_path)
        y_stats = extract_ydata_overview_stats(html_path)

    y_missing_cells = y_stats.get("missing_cells", "N/A") if y_stats else "N/A"
    y_missing_percent = y_stats.get("missing_cells_percent", "N/A") if y_stats else "N/A"
    y_dup_rows = y_stats.get("duplicate_rows", "N/A") if y_stats else "N/A"
    y_dup_percent = y_stats.get("duplicate_rows_percent", "N/A") if y_stats else "N/A"

    # ===================================================
    # LOCAL ANALYSIS
    # ===================================================
    miss_tbl = detect_missing(df)
    dup_tbl = detect_duplicates(df)
    out_tbl = detect_outliers(df)

    structural_idx = out_tbl.get("structural_indices", [])
    statistical_idx = out_tbl.get("statistical_indices", [])
    semantic_idx = out_tbl.get("semantic_indices", [])

    # ===================================================
    # MERGED OUTLIERS TABLE (WITH FILTER)
    # ===================================================
    merged_rows = []

    if statistical_idx:
        temp = df.loc[statistical_idx].copy()
        temp["__outlier_type__"] = "Statistical"
        merged_rows.append(temp)

    if semantic_idx:
        temp = df.loc[semantic_idx].copy()
        temp["__outlier_type__"] = "AI-Based"
        merged_rows.append(temp)

    if structural_idx:
        temp = df.loc[structural_idx].copy()
        temp["__outlier_type__"] = "Structural"
        merged_rows.append(temp)

    merged_df = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()

    # filter ?filter=...
    selected_filter = request.args.get("filter", "all").lower()

    if selected_filter == "statistical":
        filtered_df = merged_df[merged_df["__outlier_type__"] == "Statistical"]
    elif selected_filter == "ai":
        filtered_df = merged_df[merged_df["__outlier_type__"] == "AI-Based"]
    elif selected_filter == "structural":
        filtered_df = merged_df[merged_df["__outlier_type__"] == "Structural"]
    else:
        filtered_df = merged_df

    merged_table_html = (
        filtered_df.to_html(classes="table table-bordered table-sm", index=False)
        if not filtered_df.empty else "<p>No outliers found for this category.</p>"
    )

    # ===================================================
    # TARGET COLUMN DETECTION
    # ===================================================
    session_key = f"manual_target_column_{dataset.id}"
    manual_target = session.get(session_key)

    if manual_target and manual_target in df.columns:
        detected_target = manual_target
    else:
        detected_target = auto_detect_target_column(df)
        if manual_target and manual_target not in df.columns:
            session.pop(session_key, None)

    # ===================================================
    # LABEL ISSUES
    # ===================================================
    labels_info = detect_label_issues(df, target_col=detected_target)

    label_issue_count = labels_info.get("label_issue_count", 0)
    label_note = labels_info.get("note")
    label_preview_rows = labels_info.get("preview", [])

    label_df = pd.DataFrame(label_preview_rows)
    label_table = (
        label_df.to_html(classes="table table-bordered table-sm", index=False)
        if not label_df.empty else ""
    )

    # ===================================================
    # Missing Table
    # ===================================================
    missing_table = (
        miss_tbl.to_html(classes="table table-bordered", index=False)
        if not miss_tbl.empty else "<p>No missing values found.</p>"
    )

    # ===================================================
    # Duplicate Preview
    # ===================================================
    dup_prev_df = pd.DataFrame(dup_tbl.get("preview", []))
    dup_preview_html = (
        dup_prev_df.to_html(classes="table table-bordered table-sm", index=False)
        if not dup_prev_df.empty else "<p>No duplicate rows found.</p>"
    )

    duplicate_table = f"""
    <table class='table table-bordered table-sm'>
        <tbody>
            <tr><th>Duplicate Rows (Local)</th><td>{dup_tbl.get('duplicate_rows_count', 0)}</td></tr>
            <tr><th>Duplicate Rows (%)</th><td>{dup_tbl.get('duplicate_rows_percent', '0%')}</td></tr>
            <tr><th>Duplicate Rows (YData)</th><td>{y_dup_rows}</td></tr>
            <tr><th>Duplicate Rows (%) (YData)</th><td>{y_dup_percent}</td></tr>
        </tbody>
    </table>
    """

    # ===================================================
    # RENDER TEMPLATE
    # ===================================================
    return render_template(
        "quality_report.html",
        dataset=dataset,

        # Missing
        missing_table=missing_table,

        # Duplicates
        duplicate_table=duplicate_table,
        dup_preview_html=dup_preview_html,

        # Outliers (merged)
        merged_table=merged_table_html,
        selected_filter=selected_filter,
        outlier_total=out_tbl.get("outlier_count", 0),

        # Individual outlier counts (for header use)
        structural_count=len(structural_idx),
        statistical_count=len(statistical_idx),
        semantic_count=len(semantic_idx),

        # Labels
        label_issue_count=label_issue_count,
        label_table=label_table,
        label_note=label_note,

        # YData summary
        y_missing_cells=y_missing_cells,
        y_missing_percent=y_missing_percent,
        y_dup_rows=y_dup_rows,
        y_dup_percent=y_dup_percent,

        detected_target=detected_target,
        df_columns=df.columns.tolist(),
    )
