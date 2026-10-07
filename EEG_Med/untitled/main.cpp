#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QLabel>
#include <QMainWindow>
#include <QMessageBox>
#include <QPainter>
#include <QPointF>
#include <QPolygonF>
#include <QProcess>
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QTableWidget>
#include <QTextStream>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QWidget>
#include <QAbstractItemView>
#include <QStringList>
#include <QVector>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

enum class Modality {
    EEG,
    FNIRS
};

struct Biquad {
    double b0 = 0.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;
    double z1 = 0.0, z2 = 0.0;

    double process(double x) {
        const double y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
};

static Biquad makeButterworth(
    double samplingRate,
    double cutoff,
    bool highpass
    ) {
    constexpr double pi = 3.14159265358979323846;
    constexpr double q = 0.7071067811865476;

    const double w0 = 2.0 * pi * cutoff / samplingRate;
    const double cosine = std::cos(w0);
    const double sine = std::sin(w0);
    const double alpha = sine / (2.0 * q);

    Biquad filter;

    if (highpass) {
        filter.b0 = (1.0 + cosine) / 2.0;
        filter.b1 = -(1.0 + cosine);
        filter.b2 = (1.0 + cosine) / 2.0;
    } else {
        filter.b0 = (1.0 - cosine) / 2.0;
        filter.b1 = 1.0 - cosine;
        filter.b2 = (1.0 - cosine) / 2.0;
    }

    const double a0 = 1.0 + alpha;
    filter.b0 /= a0;
    filter.b1 /= a0;
    filter.b2 /= a0;
    filter.a1 = (-2.0 * cosine) / a0;
    filter.a2 = (1.0 - alpha) / a0;

    return filter;
}

static std::vector<double> applyFilterSection(
    const std::vector<double>& input,
    double samplingRate,
    double cutoff,
    bool highpass
    ) {
    Biquad filter = makeButterworth(samplingRate, cutoff, highpass);
    std::vector<double> output;
    output.reserve(input.size());

    for (double sample : input)
        output.push_back(filter.process(sample));

    return output;
}

// Offline demonstration band-pass filter.
static std::vector<double> bandpassFilter(
    const std::vector<double>& input,
    double samplingRate,
    double lowCut,
    double highCut
    ) {
    auto output =
        applyFilterSection(input, samplingRate, lowCut, true);
    output =
        applyFilterSection(output, samplingRate, highCut, false);

    // Forward-backward pass reduces phase shift.
    std::reverse(output.begin(), output.end());
    output =
        applyFilterSection(output, samplingRate, lowCut, true);
    output =
        applyFilterSection(output, samplingRate, highCut, false);
    std::reverse(output.begin(), output.end());

    return output;
}

class SignalPlot : public QWidget {
public:
    explicit SignalPlot(
        const QString& title,
        QWidget* parent = nullptr
        ) : QWidget(parent), title_(title) {
        setMinimumHeight(180);
    }

    void setData(
        const QVector<QPointF>& points,
        const QString& xLabel,
        const QString& yLabel
        ) {
        points_ = points;
        xLabel_ = xLabel;
        yLabel_ = yLabel;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.fillRect(rect(), Qt::white);

        const QRectF area(65, 30, width() - 85, height() - 70);

        painter.setPen(Qt::black);
        painter.drawText(
            QRectF(10, 5, width() - 20, 22),
            Qt::AlignCenter,
            title_
            );

        if (points_.size() < 2 ||
            area.width() <= 0 ||
            area.height() <= 0) {
            painter.drawText(area, Qt::AlignCenter, "No data");
            return;
        }

        double xMin = points_.first().x();
        double xMax = points_.last().x();
        double yMin = points_.first().y();
        double yMax = yMin;

        for (const QPointF& point : points_) {
            yMin = std::min(yMin, point.y());
            yMax = std::max(yMax, point.y());
        }

        if (xMax <= xMin)
            xMax = xMin + 1.0;

        if (std::abs(yMax - yMin) < 1e-12) {
            yMin -= 1.0;
            yMax += 1.0;
        }

        const double padding = 0.06 * (yMax - yMin);
        yMin -= padding;
        yMax += padding;

        painter.setPen(QPen(QColor(225, 225, 225), 1));

        for (int i = 0; i <= 4; ++i) {
            const double y = area.top() + area.height() * i / 4.0;
            painter.drawLine(
                QPointF(area.left(), y),
                QPointF(area.right(), y)
                );
        }

        painter.setPen(QPen(Qt::black, 1));
        painter.drawRect(area);

        QPolygonF line;
        line.reserve(points_.size());

        for (const QPointF& point : points_) {
            const double x = area.left() +
                             (point.x() - xMin) / (xMax - xMin) * area.width();

            const double y = area.bottom() -
                             (point.y() - yMin) / (yMax - yMin) * area.height();

            line << QPointF(x, y);
        }

        painter.setPen(QPen(QColor(30, 100, 190), 1.2));
        painter.drawPolyline(line);

        painter.setPen(Qt::black);
        painter.drawText(
            QRectF(area.left(), area.bottom() + 25, area.width(), 20),
            Qt::AlignCenter,
            xLabel_
            );

        painter.save();
        painter.translate(18, area.center().y());
        painter.rotate(-90);
        painter.drawText(
            QRectF(-area.height() / 2, -10, area.height(), 20),
            Qt::AlignCenter,
            yLabel_
            );
        painter.restore();
    }

private:
    QString title_;
    QString xLabel_;
    QString yLabel_;
    QVector<QPointF> points_;
};

struct WindowResult {
    double start = 0.0;
    double end = 0.0;
    double score = 0.0;
    double amplitudeRate = 0.0;
    double jumpRate = 0.0;
    double flatRate = 0.0;
    QString reason;
};

static QVector<QPointF> makePlotPoints(
    const std::vector<double>& data,
    double samplingRate,
    size_t maxPoints = 5000
    ) {
    QVector<QPointF> points;

    if (data.empty() || samplingRate <= 0.0)
        return points;

    const size_t stride = std::max<size_t>(1, data.size() / maxPoints);

    for (size_t i = 0; i < data.size(); i += stride) {
        points.append(
            QPointF(static_cast<double>(i) / samplingRate, data[i])
            );
    }

    return points;
}

static QVector<QPointF> makeSpectrum(
    const std::vector<double>& data,
    double samplingRate
    ) {
    const size_t n = std::min<size_t>(data.size(), 1024);
    QVector<QPointF> points;

    if (n < 16 || samplingRate <= 0.0)
        return points;

    double mean = 0.0;
    for (size_t i = 0; i < n; ++i)
        mean += data[i];
    mean /= static_cast<double>(n);

    constexpr double pi = 3.14159265358979323846;
    const size_t bins = n / 2;

    for (size_t k = 0; k < bins; ++k) {
        std::complex<double> sum(0.0, 0.0);

        for (size_t i = 0; i < n; ++i) {
            const double hann =
                0.5 - 0.5 * std::cos(
                          2.0 * pi * static_cast<double>(i) /
                          static_cast<double>(n - 1)
                          );

            const double sample = (data[i] - mean) * hann;
            const double angle =
                -2.0 * pi * static_cast<double>(k) *
                static_cast<double>(i) /
                static_cast<double>(n);

            sum += sample * std::complex<double>(
                       std::cos(angle), std::sin(angle)
                       );
        }

        const double frequency =
            static_cast<double>(k) * samplingRate /
            static_cast<double>(n);

        points.append(
            QPointF(frequency, std::abs(sum) / static_cast<double>(n))
            );
    }

    return points;
}

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    QMainWindow window;
    window.setWindowTitle("EEG / fNIRS Signal Quality Learning Tool");
    window.resize(1200, 850);

    auto* central = new QWidget;
    auto* mainLayout = new QVBoxLayout(central);
    auto* controls = new QHBoxLayout;

    auto* modalityBox = new QComboBox;
    modalityBox->addItem("EEG");
    modalityBox->addItem("fNIRS");

    auto* openButton = new QPushButton("Open CSV");
    auto* convertButton = new QPushButton("Convert EDF / SNIRF");
    auto* analyzeButton = new QPushButton("Analyze");
    auto* exportButton = new QPushButton("Export CSV");

    auto* channelBox = new QComboBox;
    auto* timeColumnBox = new QCheckBox("First column is time");

    auto* samplingRateBox = new QDoubleSpinBox;
    samplingRateBox->setRange(0.1, 10000.0);
    samplingRateBox->setDecimals(3);
    samplingRateBox->setValue(160.0);
    samplingRateBox->setSuffix(" Hz");

    auto* lowCutBox = new QDoubleSpinBox;
    lowCutBox->setRange(0.001, 100.0);
    lowCutBox->setDecimals(3);
    lowCutBox->setValue(0.5);
    lowCutBox->setSuffix(" Hz");

    auto* highCutBox = new QDoubleSpinBox;
    highCutBox->setRange(0.01, 500.0);
    highCutBox->setDecimals(3);
    highCutBox->setValue(40.0);
    highCutBox->setSuffix(" Hz");

    channelBox->setEnabled(false);
    analyzeButton->setEnabled(false);
    exportButton->setEnabled(false);

    controls->addWidget(new QLabel("Modality:"));
    controls->addWidget(modalityBox);
    controls->addWidget(openButton);
    controls->addWidget(convertButton);
    controls->addWidget(new QLabel("Channel:"));
    controls->addWidget(channelBox);
    controls->addWidget(timeColumnBox);
    controls->addWidget(new QLabel("Sampling rate:"));
    controls->addWidget(samplingRateBox);
    controls->addWidget(new QLabel("Filter:"));
    controls->addWidget(lowCutBox);
    controls->addWidget(highCutBox);
    controls->addWidget(analyzeButton);
    controls->addWidget(exportButton);

    auto* status = new QLabel(
        "EEG values should be in µV; fNIRS HbO/HbR values in µM."
        );

    auto* rawPlot = new SignalPlot("Raw signal");
    auto* filteredPlot = new SignalPlot("Filtered signal");
    auto* spectrumPlot = new SignalPlot("Signal spectrum");

    auto* topPlots = new QWidget;
    auto* plotsLayout = new QHBoxLayout(topPlots);
    plotsLayout->addWidget(rawPlot);
    plotsLayout->addWidget(filteredPlot);

    auto* table = new QTableWidget(0, 5);
    table->setHorizontalHeaderLabels({
        "Start (s)",
        "End (s)",
        "Quality index",
        "Possible issue",
        "Amplitude / jumps / flat"
    });
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);

    auto* splitter = new QSplitter(Qt::Vertical);
    splitter->addWidget(topPlots);
    splitter->addWidget(spectrumPlot);
    splitter->addWidget(table);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 2);
    splitter->setStretchFactor(2, 1);

    mainLayout->addLayout(controls);
    mainLayout->addWidget(status);
    mainLayout->addWidget(splitter);
    window.setCentralWidget(central);

    QStringList channelNames;
    QVector<QVector<double>> channels;
    std::vector<double> raw;
    std::vector<double> filtered;
    std::vector<WindowResult> windowResults;

    auto currentModality = [&]() {
        return modalityBox->currentIndex() == 0
                   ? Modality::EEG
                   : Modality::FNIRS;
    };

    auto updateModalitySettings = [&]() {
        if (currentModality() == Modality::EEG) {
            lowCutBox->setValue(0.5);
            highCutBox->setValue(40.0);
            samplingRateBox->setValue(160.0);
            rawPlot->setData({}, "Time (s)", "Amplitude (µV)");
            filteredPlot->setData({}, "Time (s)", "Amplitude (µV)");
            spectrumPlot->setData({}, "Frequency (Hz)", "Magnitude");
            status->setText(
                "EEG mode: signal values should be in microvolts (µV)."
                );
        } else {
            lowCutBox->setValue(0.01);
            highCutBox->setValue(0.2);
            samplingRateBox->setValue(10.0);
            rawPlot->setData({}, "Time (s)", "Concentration (µM)");
            filteredPlot->setData({}, "Time (s)", "Concentration (µM)");
            spectrumPlot->setData({}, "Frequency (Hz)", "Magnitude");
            status->setText(
                "fNIRS mode: HbO/HbR values should be in micromolar (µM)."
                );
        }

        channels.clear();
        channelNames.clear();
        channelBox->clear();
        channelBox->setEnabled(false);
        analyzeButton->setEnabled(false);
        exportButton->setEnabled(false);
        table->setRowCount(0);
    };

    auto analyze = [&]() {
        if (channelBox->currentIndex() < 0 || channels.isEmpty())
            return;

        const double fs = samplingRateBox->value();
        const double lowCut = lowCutBox->value();
        const double highCut = highCutBox->value();

        if (fs <= 0.0 || lowCut <= 0.0 ||
            lowCut >= highCut || highCut >= fs / 2.0) {
            QMessageBox::warning(
                &window,
                "Invalid settings",
                "Check the sampling rate and filter limits. "
                "The high cutoff must be below half the sampling rate."
                );
            return;
        }

        const auto& selected = channels[channelBox->currentIndex()];
        raw.assign(selected.begin(), selected.end());

        const size_t windowSamples = static_cast<size_t>(
            std::round(fs *
                       (currentModality() == Modality::EEG ? 1.0 : 5.0))
            );

        if (raw.size() < windowSamples) {
            QMessageBox::warning(
                &window,
                "Recording too short",
                "The selected channel is shorter than one analysis window."
                );
            return;
        }

        filtered = bandpassFilter(raw, fs, lowCut, highCut);

        const size_t count =
            (raw.size() + windowSamples - 1) / windowSamples;

        windowResults.clear();
        table->setRowCount(0);

        // Demonstration-only thresholds. They are not clinically validated.
        const double amplitudeLimit =
            currentModality() == Modality::EEG ? 150.0 : 10.0;
        const double jumpLimit =
            currentModality() == Modality::EEG ? 50.0 : 1.0;
        const double flatDifference =
            currentModality() == Modality::EEG ? 0.01 : 0.001;

        for (size_t w = 0; w < count; ++w) {
            const size_t begin = w * windowSamples;
            const size_t end = std::min(begin + windowSamples, raw.size());
            const size_t n = end - begin;

            size_t highAmplitude = 0;
            size_t jumps = 0;
            size_t flatSteps = 0;

            for (size_t i = begin; i < end; ++i) {
                if (std::abs(raw[i]) > amplitudeLimit)
                    ++highAmplitude;

                if (i > begin) {
                    const double difference =
                        std::abs(raw[i] - raw[i - 1]);

                    if (difference > jumpLimit)
                        ++jumps;

                    if (difference < flatDifference)
                        ++flatSteps;
                }
            }

            WindowResult result;
            result.start = static_cast<double>(begin) / fs;
            result.end = static_cast<double>(end) / fs;
            result.amplitudeRate =
                static_cast<double>(highAmplitude) / n;
            result.jumpRate = n > 1
                                  ? static_cast<double>(jumps) / (n - 1)
                                  : 0.0;
            result.flatRate = n > 1
                                  ? static_cast<double>(flatSteps) / (n - 1)
                                  : 0.0;

            result.score =
                0.50 * std::min(result.amplitudeRate / 0.05, 1.0) +
                0.30 * std::min(result.jumpRate / 0.01, 1.0) +
                0.20 * std::min(result.flatRate / 0.50, 1.0);

            QStringList reasons;

            if (result.amplitudeRate > 0.05)
                reasons << "high amplitude";
            if (result.jumpRate > 0.01)
                reasons << "abrupt changes";
            if (result.flatRate > 0.50)
                reasons << "nearly flat signal";

            const bool flagged = result.score >= 0.35;

            if (flagged && reasons.isEmpty())
                reasons << "combined indicators";

            result.reason = reasons.isEmpty()
                                ? "no warning"
                                : reasons.join(", ");

            windowResults.push_back(result);

            if (flagged) {
                const int row = table->rowCount();
                table->insertRow(row);

                const QStringList values = {
                    QString::number(result.start, 'f', 2),
                    QString::number(result.end, 'f', 2),
                    QString::number(result.score, 'f', 3),
                    result.reason,
                    QString("%1 / %2 / %3")
                        .arg(result.amplitudeRate, 0, 'f', 3)
                        .arg(result.jumpRate, 0, 'f', 3)
                        .arg(result.flatRate, 0, 'f', 3)
                };

                for (int col = 0; col < values.size(); ++col)
                    table->setItem(
                        row, col, new QTableWidgetItem(values[col])
                        );
            }
        }

        const QString unit = currentModality() == Modality::EEG
                                 ? "Amplitude (µV)"
                                 : "Concentration (µM)";

        rawPlot->setData(
            makePlotPoints(raw, fs), "Time (s)", unit
            );
        filteredPlot->setData(
            makePlotPoints(filtered, fs), "Time (s)", unit
            );
        spectrumPlot->setData(
            makeSpectrum(filtered, fs), "Frequency (Hz)", "Magnitude"
            );

        status->setText(
            QString("Analyzed %1 samples in %2 window(s); "
                    "%3 window(s) flagged for review. "
                    "Heuristic flags are not medical findings.")
                .arg(static_cast<qulonglong>(raw.size()))
                .arg(static_cast<qulonglong>(count))
                .arg(table->rowCount())
            );


        exportButton->setEnabled(true);
    };

    QObject::connect(
        modalityBox,
        QOverload<int>::of(&QComboBox::currentIndexChanged),
        [&](int) { updateModalitySettings(); }
        );

    QObject::connect(
        openButton,
        &QPushButton::clicked,
        [&]() {
            const QString path = QFileDialog::getOpenFileName(
                &window,
                "Open signal CSV",
                QString(),
                "CSV files (*.csv *.txt);;All files (*)"
                );

            if (path.isEmpty())
                return;

            QFile file(path);
            if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
                QMessageBox::critical(
                    &window, "Error", "Could not open the selected file."
                    );
                return;
            }

            QStringList lines;
            QTextStream in(&file);

            while (!in.atEnd()) {
                const QString line = in.readLine().trimmed();
                if (!line.isEmpty())
                    lines << line;
            }

            if (lines.isEmpty()) {
                QMessageBox::warning(
                    &window, "Empty file", "The file contains no data."
                    );
                return;
            }

            QChar separator = ',';
            const QString firstLine = lines.first();

            if (firstLine.count(';') > firstLine.count(','))
                separator = ';';
            if (firstLine.count('\t') > firstLine.count(separator))
                separator = '\t';

            const QStringList firstFields = firstLine.split(separator);

            bool hasHeader = false;
            for (const QString& field : firstFields) {
                bool ok = false;
                field.trimmed().toDouble(&ok);
                if (!ok) {
                    hasHeader = true;
                    break;
                }
            }

            const int dataStart = hasHeader ? 1 : 0;
            const int timeOffset =
                timeColumnBox->isChecked() ? 1 : 0;
            const int channelCount =
                firstFields.size() - timeOffset;

            if (channelCount <= 0) {
                QMessageBox::warning(
                    &window, "CSV format",
                    "No signal columns were found."
                    );
                return;
            }

            channelNames.clear();
            channels.clear();
            channels.resize(channelCount);

            for (int c = 0; c < channelCount; ++c) {
                if (hasHeader) {
                    channelNames << firstFields[c + timeOffset].trimmed();
                } else {
                    channelNames << QString("Channel %1").arg(c + 1);
                }
            }

            for (int row = dataStart; row < lines.size(); ++row) {
                const QStringList fields = lines[row].split(separator);

                if (fields.size() < channelCount + timeOffset)
                    continue;

                QVector<double> values;
                bool valid = true;

                for (int c = 0; c < channelCount; ++c) {
                    bool ok = false;
                    const double value =
                        fields[c + timeOffset].trimmed().toDouble(&ok);

                    if (!ok) {
                        valid = false;
                        break;
                    }
                    values.push_back(value);
                }

                if (!valid)
                    continue;

                for (int c = 0; c < channelCount; ++c)
                    channels[c].push_back(values[c]);
            }

            bool hasSamples = false;
            for (const auto& channel : channels) {
                if (!channel.isEmpty()) {
                    hasSamples = true;
                    break;
                }
            }

            if (!hasSamples) {
                QMessageBox::warning(
                    &window, "CSV format",
                    "No numeric samples found. Check the CSV format "
                    "and the time-column setting."
                    );
                channels.clear();
                return;
            }

            channelBox->clear();
            channelBox->addItems(channelNames);
            channelBox->setEnabled(true);
            analyzeButton->setEnabled(true);

            status->setText(
                QString("Loaded %1 channel(s). Check the sampling rate.")
                    .arg(channelCount)
                );

            analyze();
        }
        );

    QObject::connect(
        analyzeButton,
        &QPushButton::clicked,
        analyze
        );

    QObject::connect(
        channelBox,
        QOverload<int>::of(&QComboBox::currentIndexChanged),
        [&](int) {
            if (!channels.isEmpty())
                analyze();
        }
        );

    QObject::connect(
        convertButton,
        &QPushButton::clicked,
        [&]() {
            const bool isEeg = currentModality() == Modality::EEG;

            const QString fileFilter = isEeg
                                           ? "EEG EDF files (*.edf *.EDF)"
                                           : "fNIRS SNIRF files (*.snirf *.SNIRF)";

            const QString inputPath = QFileDialog::getOpenFileName(
                &window,
                isEeg ? "Select EEG EDF file" : "Select fNIRS SNIRF file",
                QString(),
                fileFilter
                );

            if (inputPath.isEmpty())
                return;

            const QString scriptPath =
                "C:/Users/user/Desktop/EEG_Med/untitled/edf_to_csv.py";

            if (!QFileInfo::exists(scriptPath)) {
                QMessageBox::critical(
                    &window,
                    "Script not found",
                    "Could not find edf_to_csv.py:\n" + scriptPath
                    );
                return;
            }

            const QString suggested =
                QFileInfo(inputPath).absolutePath() + "/" +
                QFileInfo(inputPath).completeBaseName() + ".csv";

            const QString outputPath = QFileDialog::getSaveFileName(
                &window,
                "Save converted CSV",
                suggested,
                "CSV files (*.csv)"
                );

            if (outputPath.isEmpty())
                return;

            auto* process = new QProcess(&window);
            process->setProcessChannelMode(QProcess::MergedChannels);

            QObject::connect(
                process,
                QOverload<int, QProcess::ExitStatus>::of(
                    &QProcess::finished
                    ),
                &window,
                [process, outputPath, &window](
                    int exitCode,
                    QProcess::ExitStatus exitStatus
                    ) {
                    const QString output =
                        QString::fromLocal8Bit(process->readAll());

                    if (exitStatus == QProcess::NormalExit &&
                        exitCode == 0) {
                        QMessageBox::information(
                            &window,
                            "Conversion complete",
                            "Converted file:\n" + outputPath +
                                "\n\nNow click Open CSV, select this file, "
                                "and enable 'First column is time'.\n\n" +
                                output
                            );
                    } else {
                        QMessageBox::critical(
                            &window,
                            "Conversion failed",
                            "Conversion failed. Check Python, MNE, "
                            "and MNE-NIRS installation.\n\n" + output
                            );
                    }

                    process->deleteLater();
                }
                );

            process->start(
                "py",
                QStringList()
                    << "-3"
                    << scriptPath
                    << inputPath
                    << outputPath
                );

            if (!process->waitForStarted(3000)) {
                QMessageBox::critical(
                    &window,
                    "Python not found",
                    "Could not start Python using the 'py' command. "
                    "Check that Python is installed."
                    );
                process->deleteLater();
            }
        }
        );

    QObject::connect(
        exportButton,
        &QPushButton::clicked,
        [&]() {
            if (raw.empty() || filtered.empty() || windowResults.empty())
                return;

            const QString path = QFileDialog::getSaveFileName(
                &window,
                "Export analyzed data",
                "signal_analysis.csv",
                "CSV files (*.csv)"
                );

            if (path.isEmpty())
                return;

            QFile file(path);
            if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                QMessageBox::critical(
                    &window, "Error", "Could not save the output file."
                    );
                return;
            }

            const bool isEeg = currentModality() == Modality::EEG;
            const QString unit = isEeg ? "uV" : "uM";

            QTextStream out(&file);
            out << "time_s,raw_" << unit
                << ",filtered_" << unit
                << ",quality_index,review_flag\n";

            const double fs = samplingRateBox->value();
            const size_t windowSeconds =
                isEeg ? 1 : 5;
            const size_t samplesPerWindow =
                std::max<size_t>(
                    1,
                    static_cast<size_t>(std::round(fs * windowSeconds))
                    );

            for (size_t i = 0; i < raw.size(); ++i) {
                const size_t w = std::min(
                    i / samplesPerWindow,
                    windowResults.size() - 1
                    );

                const bool flagged =
                    windowResults[w].score >= 0.35;

                out << QString::number(
                    static_cast<double>(i) / fs, 'f', 6
                    )
                    << ',' << QString::number(raw[i], 'f', 6)
                    << ',' << QString::number(filtered[i], 'f', 6)
                    << ',' << QString::number(
                           windowResults[w].score, 'f', 4
                           )
                    << ',' << (flagged ? "REVIEW" : "OK")
                    << '\n';
            }

            QMessageBox::information(
                &window,
                "Export complete",
                "The results have been saved."
                );
        }
        );

    window.show();
    return app.exec();
}