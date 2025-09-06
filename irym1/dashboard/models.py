from django.db import models
from sheet_models.models import Analysis

# =============================
# Descriptive Statistics
# =============================
class DescriptiveStatistics(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="descriptive_statistics")
    table = models.JSONField()

    def __str__(self):
        return f"Descriptive stats for Analysis {self.analysis.id}"

# =============================
# Correlations
# =============================
class Correlation(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="correlations")
    table = models.JSONField()

    def __str__(self):
        return f"Correlation table for Analysis {self.analysis.id}"

# =============================
# Quality Report
# =============================
class QualityReport(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="quality_reports")
    table = models.JSONField()

    def __str__(self):
        return f"Quality report for Analysis {self.analysis.id}"

# =============================
# Column Descriptions
# =============================
class ColumnDescription(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="column_desciptions")
    table = models.JSONField()

    def __str__(self):
        return f"Column descriptions for Analysis {self.analysis.id}"

# =============================
# Charts
# =============================
class Chart(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="charts")
    plot_html = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Chart for Analysis {self.analysis.id}"


"""
    In Data_analysis unit 
    How to use after Data Analysis finished
        use case:
    
            from sheet_models.models import DescriptiveStatistics

            # تحويل DataFrame لقائمة dicts
            json_data = df_stats.to_dict(orient='records')

            # إنشاء record جديد في DescriptiveStatistics
            DescriptiveStatistics.objects.create(
                analysis=analysis_instance,
                table=json_data
            )

"""

"""
    In Views
    
        from django.shortcuts import render, get_object_or_404
        from sheet_models.models import DescriptiveStatistics

        def descriptive_stats_view(request, analysis_id):
            stats = get_object_or_404(DescriptiveStatistics, analysis_id=analysis_id)
            
            # stats.table هنا عبارة عن dict أو list حسب اللي خزّنته
            table_data = stats.table  # يمكن تمريره مباشرة للـ template

            return render(request, "descriptive_stats.html", {"table_data": table_data})

"""

"""
    In Templates
    
        <table>
            <thead>
                <tr>
                {% for key in table_data.0.keys %}
                    <th>{{ key }}</th>
                {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in table_data %}
                <tr>
                    {% for value in row.values %}
                    <td>{{ value }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>

"""