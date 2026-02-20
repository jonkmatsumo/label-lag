import { useState, useEffect } from 'react';
import { Calendar } from 'lucide-react';

export interface DateRange {
    start: string;
    end: string;
}

interface DateRangePickerProps {
    onChange: (range: DateRange) => void;
    initialPreset?: '7d' | '30d' | '90d' | 'custom';
}

export function DateRangePicker({ onChange, initialPreset = '7d' }: DateRangePickerProps) {
    const [preset, setPreset] = useState<string>(initialPreset);
    const [customRange, setCustomRange] = useState<DateRange>(() => {
        const end = new Date();
        const start = new Date();
        start.setDate(end.getDate() - 7);
        return {
            start: start.toISOString().split('T')[0],
            end: end.toISOString().split('T')[0],
        };
    });

    const getRangeForPreset = (p: string): DateRange => {
        const end = new Date();
        const start = new Date();

        if (p === '7d') start.setDate(end.getDate() - 7);
        else if (p === '30d') start.setDate(end.getDate() - 30);
        else if (p === '90d') start.setDate(end.getDate() - 90);

        return {
            start: start.toISOString().split('T')[0],
            end: end.toISOString().split('T')[0],
        };
    };

    useEffect(() => {
        if (preset !== 'custom') {
            onChange(getRangeForPreset(preset));
        } else {
            onChange(customRange);
        }
    }, [preset, customRange, onChange]);

    return (
        <div className="d-flex align-items-center gap-2 flex-wrap">
            <div className="btn-group btn-group-sm">
                <button
                    type="button"
                    className={`btn ${preset === '7d' ? 'btn-primary' : 'btn-outline-secondary'}`}
                    onClick={() => setPreset('7d')}
                >
                    Last 7d
                </button>
                <button
                    type="button"
                    className={`btn ${preset === '30d' ? 'btn-primary' : 'btn-outline-secondary'}`}
                    onClick={() => setPreset('30d')}
                >
                    Last 30d
                </button>
                <button
                    type="button"
                    className={`btn ${preset === '90d' ? 'btn-primary' : 'btn-outline-secondary'}`}
                    onClick={() => setPreset('90d')}
                >
                    Last 90d
                </button>
                <button
                    type="button"
                    className={`btn ${preset === 'custom' ? 'btn-primary' : 'btn-outline-secondary'}`}
                    onClick={() => setPreset('custom')}
                >
                    Custom
                </button>
            </div>

            {preset === 'custom' && (
                <div className="d-flex align-items-center gap-2 ms-2">
                    <div className="input-group input-group-sm">
                        <span className="input-group-text bg-white border-end-0">
                            <Calendar size={14} className="text-muted" />
                        </span>
                        <input
                            type="date"
                            className="form-control border-start-0 ps-0"
                            value={customRange.start}
                            onChange={(e) => setCustomRange({ ...customRange, start: e.target.value })}
                        />
                    </div>
                    <span className="text-muted">to</span>
                    <div className="input-group input-group-sm">
                        <span className="input-group-text bg-white border-end-0">
                            <Calendar size={14} className="text-muted" />
                        </span>
                        <input
                            type="date"
                            className="form-control border-start-0 ps-0"
                            value={customRange.end}
                            onChange={(e) => setCustomRange({ ...customRange, end: e.target.value })}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
