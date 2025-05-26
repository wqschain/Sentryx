"""add token history tables

Revision ID: add_token_history_tables
Revises: # you'll need to put the ID of your last migration here
Create Date: 2024-03-21

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_token_history_tables'
down_revision = None  # Update this with your last migration ID
branch_labels = None
depends_on = None

def upgrade():
    # Add new columns to token_data table
    op.add_column('token_data', sa.Column('price_change_24h', sa.Float(), nullable=True))
    op.add_column('token_data', sa.Column('high_24h', sa.Float(), nullable=True))
    op.add_column('token_data', sa.Column('low_24h', sa.Float(), nullable=True))
    op.add_column('token_data', sa.Column('circulating_supply', sa.Float(), nullable=True))
    op.add_column('token_data', sa.Column('max_supply', sa.Float(), nullable=True))
    op.add_column('token_data', sa.Column('market_rank', sa.Integer(), nullable=True))
    op.add_column('token_data', sa.Column('market_dominance', sa.Float(), nullable=True))
    op.add_column('token_data', sa.Column('ath', sa.Float(), nullable=True))
    op.add_column('token_data', sa.Column('atl', sa.Float(), nullable=True))

    # Create token_price_history table
    op.create_table(
        'token_price_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('token_symbol', sa.String(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_token_timestamp', 'token_price_history', ['token_symbol', 'timestamp'])

    # Create token_volume_history table
    op.create_table(
        'token_volume_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('token_symbol', sa.String(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_token_volume_timestamp', 'token_volume_history', ['token_symbol', 'timestamp'])

def downgrade():
    # Drop new columns from token_data table
    op.drop_column('token_data', 'price_change_24h')
    op.drop_column('token_data', 'high_24h')
    op.drop_column('token_data', 'low_24h')
    op.drop_column('token_data', 'circulating_supply')
    op.drop_column('token_data', 'max_supply')
    op.drop_column('token_data', 'market_rank')
    op.drop_column('token_data', 'market_dominance')
    op.drop_column('token_data', 'ath')
    op.drop_column('token_data', 'atl')

    # Drop token_price_history table
    op.drop_index('idx_token_timestamp', 'token_price_history')
    op.drop_table('token_price_history')

    # Drop token_volume_history table
    op.drop_index('idx_token_volume_timestamp', 'token_volume_history')
    op.drop_table('token_volume_history') 