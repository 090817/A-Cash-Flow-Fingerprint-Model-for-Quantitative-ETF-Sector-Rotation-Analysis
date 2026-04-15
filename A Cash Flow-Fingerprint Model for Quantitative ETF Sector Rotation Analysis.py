"""
ETF行业轮动智能推荐系统
基于资金流指纹模型，结合多种因子进行行业评分与推荐
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json

# 数据获取相关库
try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False
    print("提示: baostock未安装，将使用akshare作为主要数据源")

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告: akshare未安装，请安装后重试")
    sys.exit(1)

warnings.filterwarnings('ignore')

class ETFSectorData:
    """ETF行业分类数据管理类"""
    
    # 核心行业板块ETF（精选15个）
    CORE_SECTORS = {
        "金融": "512000",      # 券商ETF
        "消费": "512600",      # 主要消费ETF
        "医药": "512010",      # 医疗ETF
        "科技": "512480",      # 半导体ETF
        "新能源": "515030",    # 新能源汽车ETF
        "光伏": "515790",      # 光伏ETF
        "军工": "512660",      # 军工ETF
        "有色金属": "512400",  # 有色金属ETF
        "煤炭": "515220",      # 煤炭ETF
        "房地产": "512200",    # 房地产ETF
        "传媒": "512980",      # 传媒ETF
        "农业": "159825",      # 农业ETF
        "计算机": "512720",    # 计算机ETF
        "5G通信": "515050",    # 5G ETF
        "基建": "516950",      # 基建ETF
    }
    
    # 宽基指数ETF
    BROAD_INDEX = {
        "沪深300": "510300",
        "上证50": "510050", 
        "中证500": "510500",
        "创业板": "159915",
        "科创50": "588000"
    }
    
    # 行业完整映射（备用）
    FULL_SECTORS = {
        "金融": ["512000", "512800", "512070"],
        "消费": ["512600", "159928", "512690"],
        "医药": ["512010", "512290", "159992"],
        "科技": ["512480", "515050", "512720"],
        "新能源": ["515030", "515790", "159755"],
        "周期": ["512400", "515220", "159870"],
        "高端制造": ["512660", "516320", "562500"],
        "其他": ["512200", "512980", "159825"]
    }
    
    @classmethod
    def get_sector_info(cls, sector_name: str) -> Dict:
        """获取行业基本信息"""
        sector_descriptions = {
            "金融": "券商、银行、保险等，对利率和宏观经济敏感",
            "消费": "食品饮料、家电、汽车等，防御性强",
            "医药": "医疗、生物医药、创新药等，成长性与防御性兼具",
            "科技": "半导体、5G、人工智能等，高成长高波动",
            "新能源": "新能源汽车、光伏、电池等，政策驱动型",
            "光伏": "光伏产业链，受能源政策影响大",
            "军工": "国防军工，政策驱动，逆周期性",
            "有色金属": "铜、铝、锂等，与大宗商品价格联动",
            "煤炭": "传统能源，高股息，周期性明显",
            "房地产": "地产开发及相关产业链",
            "传媒": "游戏、影视、广告等，消费属性",
            "农业": "种植、养殖、农产品加工",
            "计算机": "软件、云计算、信息安全",
            "5G通信": "通信设备、运营商",
            "基建": "建筑、建材、工程机械"
        }
        
        return {
            "code": cls.CORE_SECTORS.get(sector_name, ""),
            "description": sector_descriptions.get(sector_name, "暂无描述"),
            "risk_level": cls._get_risk_level(sector_name),
            "cycle_type": cls._get_cycle_type(sector_name)
        }
    
    @classmethod
    def _get_risk_level(cls, sector_name: str) -> str:
        """获取行业风险等级"""
        risk_mapping = {
            "金融": "中高", "消费": "中低", "医药": "中",
            "科技": "高", "新能源": "高", "光伏": "高",
            "军工": "中高", "有色金属": "高", "煤炭": "中高",
            "房地产": "高", "传媒": "中高", "农业": "中高",
            "计算机": "中高", "5G通信": "中高", "基建": "中"
        }
        return risk_mapping.get(sector_name, "未知")
    
    @classmethod
    def _get_cycle_type(cls, sector_name: str) -> str:
        """获取行业周期类型"""
        cycle_mapping = {
            "金融": "顺周期", "消费": "防御", "医药": "成长+防御",
            "科技": "成长", "新能源": "成长", "光伏": "成长",
            "军工": "逆周期", "有色金属": "强周期", "煤炭": "强周期",
            "房地产": "周期", "传媒": "周期", "农业": "周期",
            "计算机": "成长", "5G通信": "成长", "基建": "逆周期"
        }
        return cycle_mapping.get(sector_name, "未知")


class DataFetcher:
    """数据获取器，支持多数据源"""
    
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 数据源优先级
        self.data_sources = []
        if BAOSTOCK_AVAILABLE:
            self.data_sources.append(("baostock", self._fetch_baostock))
        self.data_sources.append(("akshare", self._fetch_akshare))
    
    def get_etf_data(self, etf_code: str, 
                     start_date: str = None,
                     end_date: str = None,
                     force_update: bool = False) -> pd.DataFrame:
        """
        获取ETF数据（多数据源尝试）
        
        Args:
            etf_code: ETF代码
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            force_update: 是否强制更新缓存
            
        Returns:
            DataFrame包含以下列：
            date, open, high, low, close, volume, amount, pct_change
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        # 检查缓存
        cache_file = os.path.join(self.cache_dir, f"{etf_code}_{start_date}_{end_date}.pkl")
        if os.path.exists(cache_file) and not force_update:
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # 检查缓存是否过期（超过1天）
                    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                    if (datetime.now() - cache_time).days < 1:
                        print(f"使用缓存数据: {etf_code}")
                        return cached_data
            except:
                pass
        
        # 尝试不同数据源
        data = None
        source_used = None
        
        for source_name, fetch_func in self.data_sources:
            try:
                print(f"尝试从 {source_name} 获取 {etf_code}...")
                data = fetch_func(etf_code, start_date, end_date)
                if data is not None and not data.empty:
                    source_used = source_name
                    print(f"  ✓ {source_name} 成功获取 {len(data)} 条记录")
                    break
            except Exception as e:
                print(f"  ✗ {source_name} 失败: {str(e)[:100]}")
                continue
        
        if data is None or data.empty:
            raise ValueError(f"所有数据源均失败: {etf_code}")
        
        # 标准化数据格式
        data = self._standardize_data(data)
        
        # 数据质量检查
        issues = self._validate_data_quality(data)
        if issues:
            print(f"数据质量警告 ({etf_code}): {', '.join(issues)}")
        
        # 保存缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass
        
        return data
    
    def _fetch_baostock(self, etf_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从baostock获取数据"""
        # 登录
        bs.login()
        
        # 确定市场前缀
        if etf_code.startswith('5'):
            bs_code = f"sh.{etf_code}"
        else:
            bs_code = f"sz.{etf_code}"
        
        # 查询数据
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 后复权
        )
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        # 登出
        bs.logout()
        
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            return df
        return pd.DataFrame()
    
    def _fetch_akshare(self, etf_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从akshare获取数据"""
        try:
            df = ak.stock_zh_a_hist(
                symbol=etf_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            return df
        except Exception as e:
            # 尝试其他akshare接口
            try:
                df = ak.fund_etf_hist_em(symbol=etf_code, period="daily")
                return df
            except:
                raise e
    
    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数据格式"""
        # 创建副本
        df = df.copy()
        
        # 重命名列（根据数据源不同）
        column_mapping = {
            '日期': 'date', 'Date': 'date', 'trade_date': 'date',
            '开盘': 'open', 'Open': 'open', 'open': 'open',
            '收盘': 'close', 'Close': 'close', 'close': 'close',
            '最高': 'high', 'High': 'high', 'high': 'high',
            '最低': 'low', 'Low': 'low', 'low': 'low',
            '成交量': 'volume', 'Volume': 'volume', 'volume': 'volume',
            '成交额': 'amount', 'Amount': 'amount', 'amount': 'amount',
            '涨跌幅': 'pct_change', '涨跌额': 'change', 'pctChg': 'pct_change'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # 确保有必要的列
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")
        
        # 转换数据类型
        df['date'] = pd.to_datetime(df['date'])
        
        # 确保数值列是数字类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        if 'amount' in df.columns:
            numeric_cols.append('amount')
        if 'pct_change' in df.columns:
            numeric_cols.append('pct_change')
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame, min_days: int = 60) -> List[str]:
        """检查数据质量"""
        issues = []
        
        # 检查数据量
        if len(df) < min_days:
            issues.append(f"数据天数不足: {len(df)} < {min_days}")
        
        # 检查缺失值
        missing_cols = df.isnull().sum()
        for col, count in missing_cols.items():
            if count > 0:
                issues.append(f"{col}有{count}个缺失值")
        
        # 检查异常值（价格为零或负值）
        if (df['close'] <= 0).any():
            issues.append("存在异常收盘价")
        
        # 检查成交量（过多零成交量）
        zero_volume_ratio = (df['volume'] == 0).sum() / len(df)
        if zero_volume_ratio > 0.1:
            issues.append(f"零成交量比例过高: {zero_volume_ratio:.1%}")
        
        return issues
    
    def get_all_sectors_data(self, sectors_dict: Dict = None, 
                            days_back: int = 180) -> Dict[str, pd.DataFrame]:
        """
        获取所有板块数据
        
        Args:
            sectors_dict: 板块字典，默认使用核心板块
            days_back: 回溯天数
            
        Returns:
            字典: {板块名: DataFrame}
        """
        if sectors_dict is None:
            sectors_dict = ETFSectorData.CORE_SECTORS
        
        all_data = {}
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
        
        print(f"开始获取{len(sectors_dict)}个板块数据...")
        print(f"时间范围: {start_date} 至 {end_date}")
        print("-" * 50)
        
        for sector_name, etf_code in sectors_dict.items():
            try:
                data = self.get_etf_data(etf_code, start_date, end_date)
                if not data.empty:
                    data['sector'] = sector_name
                    data['etf_code'] = etf_code
                    all_data[sector_name] = data
                    print(f"✓ {sector_name:10} {etf_code} - {len(data)}条记录")
                else:
                    print(f"✗ {sector_name:10} {etf_code} - 数据为空")
            except Exception as e:
                print(f"✗ {sector_name:10} {etf_code} - 获取失败: {str(e)[:50]}")
        
        print("-" * 50)
        print(f"数据获取完成，成功: {len(all_data)}/{len(sectors_dict)}")
        
        return all_data


class FlowFingerprintModel:
    """资金流指纹模型"""
    
    def __init__(self, config: Dict = None):
        """
        初始化模型
        
        Args:
            config: 模型配置参数
        """
        # 默认配置
        self.config = {
            # 时间窗口参数
            'return_window': 5,      # 收益率窗口
            'volume_window': 5,      # 成交量窗口
            'position_window': 20,   # 价格位置窗口
            'trend_window': 30,      # 趋势窗口
            
            # 资金流因子权重（总权重需>30%）
            'weights': {
                'momentum': 0.25,    # 动量因子（价格相关）
                'flow': 0.35,        # 资金流因子（核心，>30%）
                'position': 0.20,    # 位置因子
                'trend': 0.15,       # 趋势因子
                'stability': 0.05,   # 稳定性因子
            },
            
            # 波动性参数
            'noise_threshold': 0.15,  # 人为因素造成的波动阈值
            'max_signal_change': 0.3, # 最大信号变化率（避免跳变）
            
            # 回测参数
            'backtest_periods': [5, 10, 20],  # 回测周期
        }
        
        if config:
            self.config.update(config)
        
        # 验证权重
        self._validate_weights()
    
    def _validate_weights(self):
        """验证权重配置"""
        weights = self.config['weights']
        total_weight = sum(weights.values())
        
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"权重总和应为1.0，实际为{total_weight}")
        
        # 确保资金流因子权重>30%
        if weights['flow'] < 0.3:
            raise ValueError(f"资金流因子权重应大于30%，当前为{weights['flow']*100}%")
    
    def calculate_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子
        
        Args:
            data: 包含基础行情数据的DataFrame
            
        Returns:
            包含所有因子的DataFrame
        """
        df = data.copy()
        
        # 1. 动量因子
        df['return_5d'] = df['close'].pct_change(self.config['return_window'])
        df['return_10d'] = df['close'].pct_change(10)
        
        # 相对强度（相对于自身平均）
        df['return_5d_sma'] = df['return_5d'].rolling(10).mean()
        df['rel_strength'] = df['return_5d'] - df['return_5d_sma']
        
        # 2. 资金流因子（核心）
        if 'amount' in df.columns:
            df['amount_ma5'] = df['amount'].rolling(self.config['volume_window']).mean()
            df['flow_change'] = (df['amount'] / df['amount_ma5']) - 1
            
            # 成交额突破度
            df['amount_max20'] = df['amount'].rolling(20).max()
            df['flow_breakthrough'] = df['amount'] / df['amount_max20']
            
            # 量价配合度（价格涨时量增，跌时量缩为佳）
            df['flow_price_alignment'] = df['return_5d'] * df['flow_change']
        else:
            # 如果没有成交额数据，用成交量代替
            df['volume_ma5'] = df['volume'].rolling(self.config['volume_window']).mean()
            df['flow_change'] = (df['volume'] / df['volume_ma5']) - 1
            df['flow_price_alignment'] = df['return_5d'] * df['flow_change']
        
        # 3. 位置因子
        df['high_20'] = df['close'].rolling(self.config['position_window']).max()
        df['low_20'] = df['close'].rolling(self.config['position_window']).min()
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        # 避免除零错误
        df['price_position'] = df['price_position'].fillna(0.5)
        df['price_position'] = df['price_position'].clip(0, 1)
        
        # 4. 趋势因子
        df['ma_short'] = df['close'].rolling(5).mean()
        df['ma_medium'] = df['close'].rolling(20).mean()
        df['ma_long'] = df['close'].rolling(60).mean()
        
        df['trend_strength'] = (df['ma_short'] / df['ma_long'] - 1)
        
        # 5. 稳定性因子
        df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
        df['stability_score'] = 1 / (1 + df['volatility_20d'])
        
        # 6. 考虑人为因素造成的波动（噪音过滤）
        df = self._add_noise_filter(df)
        
        return df
    
    def _add_noise_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加噪音过滤，考虑人为因素造成的异常波动
        
        Args:
            df: 包含因子的DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 检测异常波动（涨跌幅过大但成交量不配合）
        df['abnormal_move'] = 0
        
        # 条件1: 单日涨跌幅过大
        if 'pct_change' in df.columns:
            large_move = abs(df['pct_change']) > self.config['noise_threshold']
            
            # 条件2: 成交量变化不大（可能是个别大单造成）
            if 'flow_change' in df.columns:
                low_volume_change = abs(df['flow_change']) < 0.1  # 成交量变化<10%
                df.loc[large_move & low_volume_change, 'abnormal_move'] = 1
        
        # 平滑处理（避免信号跳变）
        for col in ['flow_change', 'return_5d', 'flow_price_alignment']:
            if col in df.columns:
                # 使用移动平均平滑
                df[f'{col}_smoothed'] = df[col].rolling(3, min_periods=1).mean()
        
        return df
    
    def normalize_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化因子值到0-100分
        
        Args:
            df: 包含因子的DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        norm_df = df.copy()
        
        # 需要标准化的因子列
        factor_columns = [
            'return_5d', 'rel_strength', 'flow_change', 
            'flow_breakthrough', 'flow_price_alignment',
            'price_position', 'trend_strength', 'stability_score'
        ]
        
        # 只处理存在的列
        available_cols = [col for col in factor_columns if col in norm_df.columns]
        
        for col in available_cols:
            # 去除极端值
            q_low = norm_df[col].quantile(0.05)
            q_high = norm_df[col].quantile(0.95)
            clipped = norm_df[col].clip(q_low, q_high)
            
            # 线性映射到0-100
            min_val = clipped.min()
            max_val = clipped.max()
            
            if max_val > min_val:  # 避免除零
                norm_df[f'{col}_score'] = 100 * (clipped - min_val) / (max_val - min_val)
            else:
                norm_df[f'{col}_score'] = 50  # 中间值
        
        return norm_df
    
    def calculate_fingerprint(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算完整的资金流指纹
        
        Args:
            data: 基础行情数据
            
        Returns:
            包含指纹分数的DataFrame
        """
        # 计算因子
        factor_df = self.calculate_factors(data)
        
        # 标准化因子
        norm_df = self.normalize_factors(factor_df)
        
        # 计算综合得分
        weights = self.config['weights']
        
        # 动量维度
        momentum_score = 0
        if 'return_5d_score' in norm_df.columns:
            momentum_score += norm_df['return_5d_score'] * 0.7
        if 'rel_strength_score' in norm_df.columns:
            momentum_score += norm_df['rel_strength_score'] * 0.3
        
        # 资金流维度（核心）
        flow_score = 0
        flow_components = []
        
        if 'flow_change_score' in norm_df.columns:
            flow_components.append(('flow_change', 0.4))
        if 'flow_price_alignment_score' in norm_df.columns:
            flow_components.append(('flow_price_alignment', 0.4))
        if 'flow_breakthrough_score' in norm_df.columns:
            flow_components.append(('flow_breakthrough', 0.2))
        
        if flow_components:
            for col, weight in flow_components:
                flow_score += norm_df[f'{col}_score'] * weight
        else:
            flow_score = 50  # 默认值
        
        # 位置维度
        position_score = norm_df.get('price_position_score', 50)
        
        # 趋势维度
        trend_score = norm_df.get('trend_strength_score', 50)
        
        # 稳定性维度
        stability_score = norm_df.get('stability_score', 50)
        
        # 综合得分（加权平均）
        norm_df['fingerprint_score'] = (
            momentum_score * weights['momentum'] +
            flow_score * weights['flow'] +
            position_score * weights['position'] +
            trend_score * weights['trend'] +
            stability_score * weights['stability']
        )
        
        # 添加详细信息
        norm_df['momentum_component'] = momentum_score
        norm_df['flow_component'] = flow_score
        norm_df['position_component'] = position_score
        norm_df['trend_component'] = trend_score
        norm_df['stability_component'] = stability_score
        
        return norm_df
    
    def analyze_sector(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        分析所有板块，生成排名
        
        Args:
            data_dict: 板块数据字典
            
        Returns:
            包含分析结果的DataFrame
        """
        results = []
        
        for sector_name, data in data_dict.items():
            if len(data) < 30:  # 至少需要30天数据
                continue
            
            # 计算指纹
            fingerprint_df = self.calculate_fingerprint(data)
            
            if fingerprint_df.empty:
                continue
            
            # 取最新数据
            latest = fingerprint_df.iloc[-1]
            
            # 计算近期表现
            recent_window = 5
            if len(fingerprint_df) >= recent_window:
                recent_score = fingerprint_df['fingerprint_score'].tail(recent_window).mean()
                recent_volatility = fingerprint_df['fingerprint_score'].tail(recent_window).std()
            else:
                recent_score = latest['fingerprint_score']
                recent_volatility = 0
            
            # 计算历史稳定性
            historical_scores = fingerprint_df['fingerprint_score']
            historical_mean = historical_scores.mean()
            historical_std = historical_scores.std()
            
            # 稳定性评分（标准差越小越稳定）
            stability_rating = 100 / (1 + historical_std)
            
            # 趋势判断
            score_trend = "上升" if recent_score > historical_mean else "下降"
            
            # 收集结果
            result = {
                '板块': sector_name,
                'ETF代码': data['etf_code'].iloc[0],
                '综合分数': round(latest['fingerprint_score'], 2),
                '近期分数': round(recent_score, 2),
                '资金流贡献': round(latest['flow_component'], 2),
                '趋势强度': round(latest['trend_component'], 2),
                '价格位置': round(latest['position_component'], 2),
                '稳定性': round(stability_rating, 2),
                '分数趋势': score_trend,
                '最新收盘价': round(latest['close'], 3),
                '5日涨跌幅%': round(latest['return_5d'] * 100, 2),
                '分析日期': latest['date'].strftime('%Y-%m-%d')
            }
            
            results.append(result)
        
        # 创建DataFrame并排序
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('综合分数', ascending=False)
            results_df.reset_index(drop=True, inplace=True)
            results_df.index = results_df.index + 1  # 从1开始编号
            
            return results_df
        else:
            return pd.DataFrame()


class InvestmentRecommender:
    """投资建议生成器"""
    
    def __init__(self, sector_data: ETFSectorData):
        self.sector_data = sector_data
    
    def generate_recommendations(self, analysis_df: pd.DataFrame, 
                               top_n: int = 3) -> Dict:
        """
        生成投资建议
        
        Args:
            analysis_df: 分析结果DataFrame
            top_n: 推荐前N个板块
            
        Returns:
            包含建议的字典
        """
        if analysis_df.empty:
            return {"error": "无有效分析数据"}
        
        # 获取前N个板块
        top_sectors = analysis_df.head(top_n)
        
        recommendations = {
            "生成时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "数据周期": f"最近{len(analysis_df)}个板块",
            "推荐策略": "基于资金流指纹的行业轮动",
            "核心逻辑": "资金流向是股价变动的先行指标，结合价格位置和趋势判断",
            "推荐板块": []
        }
        
        for idx, row in top_sectors.iterrows():
            sector_name = row['板块']
            sector_info = self.sector_data.get_sector_info(sector_name)
            
            # 生成投资逻辑
            investment_logic = self._generate_logic(row)
            
            # 风险评估
            risk_assessment = self._assess_risk(row, sector_info['risk_level'])
            
            recommendation = {
                "排名": int(idx),
                "板块名称": sector_name,
                "ETF代码": row['ETF代码'],
                "综合评分": row['综合分数'],
                "资金流得分": row['资金流贡献'],
                "投资逻辑": investment_logic,
                "适合投资者": self._get_investor_type(row, sector_info),
                "建议仓位": self._get_position_suggestion(idx, row['稳定性']),
                "止损建议": f"跌破前低{max(5, 15 - row['稳定性']/10):.0f}%考虑止损",
                "持有周期": self._get_holding_period(row['趋势强度']),
                "风险评估": risk_assessment,
                "行业描述": sector_info['description'],
                "风险等级": sector_info['risk_level'],
                "周期类型": sector_info['cycle_type']
            }
            
            recommendations["推荐板块"].append(recommendation)
        
        # 总体建议
        recommendations["总体建议"] = self._generate_overall_advice(top_sectors)
        
        return recommendations
    
    def _generate_logic(self, row: pd.Series) -> str:
        """生成投资逻辑"""
        logic_parts = []
        
        # 资金流逻辑
        if row['资金流贡献'] > 70:
            logic_parts.append("资金大幅流入，市场关注度高")
        elif row['资金流贡献'] > 50:
            logic_parts.append("资金温和流入，有增量资金")
        
        # 位置逻辑
        if row['价格位置'] > 70:
            logic_parts.append("价格处于相对高位，注意风险")
        elif row['价格位置'] < 30:
            logic_parts.append("价格处于相对低位，安全边际较高")
        else:
            logic_parts.append("价格位置适中，有一定上涨空间")
        
        # 趋势逻辑
        if row['趋势强度'] > 60:
            logic_parts.append("趋势明确向上")
        elif row['趋势强度'] < 40:
            logic_parts.append("趋势较弱或向下")
        
        # 稳定性逻辑
        if row['稳定性'] > 70:
            logic_parts.append("走势相对稳健")
        
        return "；".join(logic_parts)
    
    def _get_investor_type(self, row: pd.Series, sector_info: Dict) -> str:
        """确定适合的投资者类型"""
        risk_level = sector_info['risk_level']
        stability = row['稳定性']
        
        if risk_level == "高" or stability < 50:
            return "激进型投资者"
        elif risk_level == "中高":
            return "成长型投资者"
        else:
            return "稳健型投资者"
    
    def _get_position_suggestion(self, rank: int, stability: float) -> str:
        """获取仓位建议"""
        base_position = max(10, 30 - (rank-1)*5)  # 排名越靠前，建议仓位越高
        
        # 根据稳定性调整
        if stability > 70:
            adjustment = 5
        elif stability < 50:
            adjustment = -5
        else:
            adjustment = 0
        
        final_position = base_position + adjustment
        
        return f"总仓位的{final_position}%"
    
    def _get_holding_period(self, trend_strength: float) -> str:
        """获取持有周期建议"""
        if trend_strength > 70:
            return "1-3个月（趋势较强）"
        elif trend_strength > 50:
            return "2-4周（趋势中等）"
        else:
            return "1-2周（短线交易）"
    
    def _assess_risk(self, row: pd.Series, base_risk: str) -> Dict:
        """风险评估"""
        risks = []
        mitigations = []
        
        # 价格位置风险
        if row['价格位置'] > 80:
            risks.append("价格处于历史高位")
            mitigations.append("分批建仓，设置严格止损")
        
        # 稳定性风险
        if row['稳定性'] < 50:
            risks.append("波动较大")
            mitigations.append("减小仓位，增加交易频率")
        
        # 趋势风险
        if row['趋势强度'] < 40:
            risks.append("趋势不明确")
            mitigations.append("等待趋势确认再入场")
        
        return {
            "风险等级": base_risk,
            "主要风险点": risks if risks else ["风险相对可控"],
            "风控措施": mitigations if mitigations else ["正常持仓管理"]
        }
    
    def _generate_overall_advice(self, top_sectors: pd.DataFrame) -> List[str]:
        """生成总体建议"""
        advice = [
            "1. 建议分散投资2-3个不同行业的ETF，降低单一行业风险",
            "2. 排名靠前的板块资金流入明显，可作为重点配置",
            "3. 定期（每周）更新排名，根据市场变化调整持仓",
            "4. 设置止损位，控制单笔交易最大亏损在5%以内",
            "5. 结合大盘走势，在上升趋势中增加仓位，下跌趋势中减少仓位"
        ]
        
        # 根据板块特点添加建议
        sectors_list = top_sectors['板块'].tolist()
        
        if any(s in sectors_list for s in ["科技", "新能源", "光伏"]):
            advice.append("6. 科技成长板块波动较大，建议采用分批买入策略")
        
        if any(s in sectors_list for s in ["消费", "医药"]):
            advice.append("7. 消费医药板块防御性较强，适合作为底仓配置")
        
        if any(s in sectors_list for s in ["金融", "房地产", "基建"]):
            advice.append("8. 金融周期板块对经济政策敏感，关注政策动向")
        
        return advice


class ModelValidator:
    """模型验证器"""
    
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
    
    def validate_formula(self, model: FlowFingerprintModel) -> Dict:
        """验证公式合理性"""
        print("正在验证模型公式...")
        
        issues = []
        warnings = []
        
        # 1. 检查权重配置
        weights = model.config['weights']
        if weights['flow'] < 0.3:
            issues.append(f"资金流因子权重({weights['flow']*100}%)低于30%要求")
        
        # 2. 检查参数范围
        if model.config['noise_threshold'] > 0.3:
            warnings.append(f"噪音阈值({model.config['noise_threshold']})偏高")
        
        # 3. 检查时间窗口
        windows = [model.config['return_window'], model.config['volume_window'], 
                  model.config['position_window']]
        if any(w < 3 for w in windows):
            issues.append("时间窗口参数过小，可能导致过度敏感")
        
        # 测试数据验证
        test_data = self._get_test_data()
        if test_data is not None:
            try:
                result = model.calculate_fingerprint(test_data)
                
                if 'fingerprint_score' not in result.columns:
                    issues.append("指纹分数计算失败")
                else:
                    # 检查分数范围
                    scores = result['fingerprint_score']
                    if scores.min() < 0 or scores.max() > 100:
                        warnings.append("指纹分数超出0-100范围")
                    
                    # 检查稳定性
                    score_std = scores.std()
                    if score_std > 20:
                        warnings.append(f"指纹分数波动较大(标准差={score_std:.2f})")
            
            except Exception as e:
                issues.append(f"模型计算测试失败: {str(e)}")
        
        validation_result = {
            "通过": len(issues) == 0,
            "问题": issues,
            "警告": warnings,
            "建议": self._generate_suggestions(issues, warnings)
        }
        
        return validation_result
    
    def _get_test_data(self) -> Optional[pd.DataFrame]:
        """获取测试数据"""
        try:
            # 使用沪深300作为测试数据
            test_data = self.data_fetcher.get_etf_data("510300", days_back=30)
            return test_data
        except:
            return None
    
    def _generate_suggestions(self, issues: List[str], warnings: List[str]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if issues:
            suggestions.append("请优先解决上述问题")
        
        if warnings:
            suggestions.append("建议关注上述警告，优化参数配置")
        
        if not issues and not warnings:
            suggestions.append("模型公式验证通过，可以正常使用")
        
        return suggestions
    
    def check_overfitting(self, model: FlowFingerprintModel, 
                         sectors_dict: Dict) -> Dict:
        """检查过拟合"""
        print("正在检查过拟合风险...")
        
        # 使用不同时间段的数据进行验证
        time_periods = [
            ("近期", 30),   # 最近30天
            ("中期", 90),   # 最近90天
            ("长期", 180)   # 最近180天
        ]
        
        results = {}
        
        for period_name, days in time_periods:
            try:
                # 获取数据
                all_data = {}
                for sector_name, etf_code in list(sectors_dict.items())[:3]:  # 只测试前3个
                    try:
                        data = self.data_fetcher.get_etf_data(
                            etf_code, 
                            days_back=days
                        )
                        if not data.empty:
                            all_data[sector_name] = data
                    except:
                        continue
                
                if len(all_data) >= 2:  # 至少需要2个板块
                    # 分析排名
                    analysis_df = model.analyze_sector(all_data)
                    
                    if not analysis_df.empty:
                        # 获取排名
                        rankings = analysis_df[['板块', '综合分数']].to_dict('records')
                        results[period_name] = {
                            "板块数量": len(analysis_df),
                            "平均分数": analysis_df['综合分数'].mean(),
                            "分数标准差": analysis_df['综合分数'].std(),
                            "排名": rankings
                        }
            
            except Exception as e:
                results[period_name] = {"错误": str(e)}
        
        # 分析一致性
        consistency_score = self._calculate_consistency(results)
        
        return {
            "时间段表现": results,
            "一致性评分": consistency_score,
            "过拟合风险评估": self._assess_overfitting_risk(consistency_score)
        }
    
    def _calculate_consistency(self, results: Dict) -> float:
        """计算不同时间段排名一致性"""
        if len(results) < 2:
            return 0.0
        
        # 提取各时间段排名前3的板块
        top_sectors_per_period = {}
        for period_name, period_data in results.items():
            if "排名" in period_data:
                rankings = period_data["排名"]
                top_sectors = [r["板块"] for r in rankings[:3]]  # 取前3名
                top_sectors_per_period[period_name] = top_sectors
        
        # 计算重叠度
        if len(top_sectors_per_period) >= 2:
            periods = list(top_sectors_per_period.keys())
            overlap_scores = []
            
            for i in range(len(periods)):
                for j in range(i+1, len(periods)):
                    set1 = set(top_sectors_per_period[periods[i]])
                    set2 = set(top_sectors_per_period[periods[j]])
                    
                    if set1 and set2:
                        overlap = len(set1.intersection(set2)) / len(set1.union(set2))
                        overlap_scores.append(overlap)
            
            if overlap_scores:
                return np.mean(overlap_scores)
        
        return 0.0
    
    def _assess_overfitting_risk(self, consistency_score: float) -> Dict:
        """评估过拟合风险"""
        if consistency_score > 0.6:
            risk_level = "低"
            assessment = "模型在不同时间段表现一致，过拟合风险较低"
        elif consistency_score > 0.3:
            risk_level = "中"
            assessment = "模型有一定的一致性，建议进一步验证"
        else:
            risk_level = "高"
            assessment = "模型在不同时间段表现差异较大，可能存在过拟合"
        
        return {
            "风险等级": risk_level,
            "评估": assessment,
            "一致性分数": round(consistency_score, 3),
            "建议": "建议增加验证数据集和交叉验证" if risk_level == "高" else "可继续使用"
        }


class ETFSectorRecommendationSystem:
    """ETF行业轮动推荐系统主类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化推荐系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.sector_data = ETFSectorData()
        self.data_fetcher = DataFetcher(cache_dir=self.config.get('cache_dir', './data_cache'))
        self.model = FlowFingerprintModel(self.config.get('model_config'))
        self.recommender = InvestmentRecommender(self.sector_data)
        self.validator = ModelValidator(self.data_fetcher)
        
        # 状态跟踪
        self.last_update = None
        self.current_rankings = None
        self.current_recommendations = None
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        default_config = {
            'cache_dir': './data_cache',
            'days_back': 180,
            'top_n_recommendations': 3,
            'enable_validation': True,
            'model_config': {
                'return_window': 5,
                'volume_window': 5,
                'position_window': 20,
                'weights': {
                    'momentum': 0.25,
                    'flow': 0.35,
                    'position': 0.20,
                    'trend': 0.15,
                    'stability': 0.05,
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 合并配置
                default_config.update(user_config)
            except Exception as e:
                print(f"配置文件加载失败: {e}")
        
        return default_config
    
    def run_analysis(self, force_update: bool = False) -> bool:
        """
        运行完整分析流程
        
        Args:
            force_update: 是否强制更新数据
            
        Returns:
            是否成功
        """
        print("=" * 60)
        print("ETF行业轮动智能推荐系统")
        print("=" * 60)
        
        try:
            # 1. 数据获取
            print("\n1. 数据获取阶段...")
            all_data = self.data_fetcher.get_all_sectors_data(
                self.sector_data.CORE_SECTORS,
                days_back=self.config['days_back']
            )
            
            if len(all_data) < 5:
                print("错误: 获取到的有效数据太少")
                return False
            
            # 2. 模型验证（可选）
            if self.config.get('enable_validation', True):
                print("\n2. 模型验证阶段...")
                
                # 公式验证
                formula_result = self.validator.validate_formula(self.model)
                if not formula_result["通过"]:
                    print("公式验证失败:")
                    for issue in formula_result["问题"]:
                        print(f"  - {issue}")
                
                # 过拟合检查
                overfitting_result = self.validator.check_overfitting(
                    self.model, 
                    self.sector_data.CORE_SECTORS
                )
                print(f"过拟合风险评估: {overfitting_result['过拟合风险评估']['风险等级']}")
            
            # 3. 模型计算
            print("\n3. 模型计算阶段...")
            self.current_rankings = self.model.analyze_sector(all_data)
            
            if self.current_rankings.empty:
                print("错误: 模型计算失败")
                return False
            
            # 4. 生成建议
            print("\n4. 生成投资建议...")
            self.current_recommendations = self.recommender.generate_recommendations(
                self.current_rankings,
                top_n=self.config.get('top_n_recommendations', 3)
            )
            
            # 5. 更新状态
            self.last_update = datetime.now()
            
            print("\n✓ 分析完成!")
            return True
            
        except Exception as e:
            print(f"\n✗ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def display_results(self):
        """显示分析结果"""
        if self.current_rankings is None:
            print("请先运行分析流程 (run_analysis)")
            return
        
        print("\n" + "=" * 60)
        print("行业排名结果")
        print("=" * 60)
        
        # 显示排名表格
        display_df = self.current_rankings.copy()
        
        # 美化显示
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 15)
        
        # 添加颜色标记
        def color_score(val):
            if val >= 70:
                return f'\033[92m{val:.1f}\033[0m'  # 绿色
            elif val >= 50:
                return f'\033[93m{val:.1f}\033[0m'  # 黄色
            else:
                return f'\033[91m{val:.1f}\033[0m'  # 红色
        
        # 应用颜色
        display_df['综合分数'] = display_df['综合分数'].apply(color_score)
        display_df['资金流贡献'] = display_df['资金流贡献'].apply(color_score)
        
        print(display_df.to_string())
        
        # 显示投资建议
        if self.current_recommendations:
            print("\n" + "=" * 60)
            print("投资建议")
            print("=" * 60)
            
            recs = self.current_recommendations
            print(f"生成时间: {recs['生成时间']}")
            print(f"核心逻辑: {recs['核心逻辑']}")
            
            for rec in recs['推荐板块']:
                print(f"\n【第{rec['排名']}名】{rec['板块名称']} ({rec['ETF代码']})")
                print(f"  综合评分: {rec['综合评分']} (资金流: {rec['资金流得分']})")
                print(f"  投资逻辑: {rec['投资逻辑']}")
                print(f"  适合投资者: {rec['适合投资者']}")
                print(f"  建议仓位: {rec['建议仓位']}")
                print(f"  持有周期: {rec['持有周期']}")
                print(f"  风险评估: {rec['风险评估']['风险等级']} - {rec['风险评估']['主要风险点'][0]}")
            
            print("\n" + "-" * 60)
            print("总体建议:")
            for advice in recs['总体建议']:
                print(f"  {advice}")
    
    def save_results(self, output_dir: str = "./output"):
        """保存结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存排名数据
        if self.current_rankings is not None:
            csv_path = os.path.join(output_dir, f"rankings_{timestamp}.csv")
            self.current_rankings.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"排名数据已保存: {csv_path}")
        
        # 保存建议数据
        if self.current_recommendations is not None:
            json_path = os.path.join(output_dir, f"recommendations_{timestamp}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_recommendations, f, ensure_ascii=False, indent=2)
            print(f"投资建议已保存: {json_path}")
        
        # 保存摘要报告
        self._save_summary_report(output_dir, timestamp)
    
    def _save_summary_report(self, output_dir: str, timestamp: str):
        """保存摘要报告"""
        if self.current_rankings is None or self.current_recommendations is None:
            return
        
        report_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ETF行业轮动分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据回溯天数: {self.config['days_back']}\n")
            f.write(f"分析板块数量: {len(self.current_rankings)}\n\n")
            
            f.write("【行业排名TOP5】\n")
            f.write("-" * 60 + "\n")
            for idx, row in self.current_rankings.head(5).iterrows():
                f.write(f"{idx}. {row['板块']:10} - 分数: {row['综合分数']:.1f} "
                       f"(资金流: {row['资金流贡献']:.1f}, 趋势: {row['趋势强度']:.1f})\n")
            
            f.write("\n【投资建议】\n")
            f.write("-" * 60 + "\n")
            recs = self.current_recommendations
            for rec in recs['推荐板块']:
                f.write(f"\n{rec['排名']}. {rec['板块名称']} ({rec['ETF代码']})\n")
                f.write(f"   建议仓位: {rec['建议仓位']}\n")
                f.write(f"   持有周期: {rec['持有周期']}\n")
                f.write(f"   风险等级: {rec['风险评估']['风险等级']}\n")
                f.write(f"   投资逻辑: {rec['投资逻辑']}\n")
            
            f.write("\n【风险提示】\n")
            f.write("-" * 60 + "\n")
            f.write("1. 本报告基于历史数据计算，不构成投资建议\n")
            f.write("2. 市场有风险，投资需谨慎\n")
            f.write("3. 建议结合自身风险承受能力做出投资决策\n")
            f.write("4. 定期更新分析，适应市场变化\n")
        
        print(f"摘要报告已保存: {report_path}")
    
    def get_sector_details(self, sector_name: str) -> Dict:
        """获取指定行业的详细信息"""
        # 获取基础信息
        info = self.sector_data.get_sector_info(sector_name)
        
        # 获取历史数据
        etf_code = self.sector_data.CORE_SECTORS.get(sector_name)
        if not etf_code:
            return {"error": f"未找到行业: {sector_name}"}
        
        try:
            data = self.data_fetcher.get_etf_data(etf_code, days_back=30)
            
            if data.empty:
                info['data_status'] = "数据获取失败"
                return info
            
            # 计算近期表现
            latest_close = data['close'].iloc[-1]
            week_ago_close = data['close'].iloc[-5] if len(data) >= 5 else latest_close
            month_ago_close = data['close'].iloc[-20] if len(data) >= 20 else latest_close
            
            weekly_return = (latest_close / week_ago_close - 1) * 100
            monthly_return = (latest_close / month_ago_close - 1) * 100
            
            info.update({
                'data_status': "数据正常",
                'latest_price': round(latest_close, 3),
                'weekly_return': round(weekly_return, 2),
                'monthly_return': round(monthly_return, 2),
                'data_points': len(data),
                'data_period': f"{data['date'].iloc[0].strftime('%Y-%m-%d')} 至 "
                              f"{data['date'].iloc[-1].strftime('%Y-%m-%d')}"
            })
            
            return info
            
        except Exception as e:
            info['data_status'] = f"数据获取失败: {str(e)[:50]}"
            return info


def main():
    """主函数"""
    print("正在启动ETF行业轮动推荐系统...")
    
    try:
        # 创建推荐系统实例
        system = ETFSectorRecommendationSystem()
        
        # 运行分析
        success = system.run_analysis(force_update=False)
        
        if success:
            # 显示结果
            system.display_results()
            
            # 保存结果
            system.save_results()
            
            # 示例：获取特定行业详情
            print("\n" + "=" * 60)
            print("行业详情示例（科技板块）:")
            print("=" * 60)
            tech_details = system.get_sector_details("科技")
            for key, value in tech_details.items():
                print(f"{key}: {value}")
        
        else:
            print("分析失败，请检查网络连接和数据源")
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


def quick_test():
    """快速测试函数"""
    print("进行快速测试...")
    
    # 创建数据获取器
    fetcher = DataFetcher()
    
    # 测试获取单个ETF数据
    try:
        test_data = fetcher.get_etf_data("510300", days_back=30)
        print(f"测试数据获取成功，记录数: {len(test_data)}")
        
        # 创建模型
        model = FlowFingerprintModel()
        
        # 计算指纹
        fingerprint = model.calculate_fingerprint(test_data)
        print(f"指纹计算成功，最新分数: {fingerprint['fingerprint_score'].iloc[-1]:.2f}")
        
        # 验证模型
        validator = ModelValidator(fetcher)
        formula_result = validator.validate_formula(model)
        print(f"公式验证: {'通过' if formula_result['通过'] else '失败'}")
        
        return True
        
    except Exception as e:
        print(f"快速测试失败: {e}")
        return False


if __name__ == "__main__":
    # 检查依赖
    if not AKSHARE_AVAILABLE:
        print("请安装akshare: pip install akshare")
        sys.exit(1)
    
    if not BAOSTOCK_AVAILABLE:
        print("提示: 未安装baostock，将仅使用akshare作为数据源")
        print("      如需安装: pip install baostock")
    
    # 运行主程序
    main()
    
    # 可选：运行快速测试
    # quick_test()